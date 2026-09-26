package service

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/httpclient"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const codexTimezoneFallback = "Asia/Singapore"

// Keep the reference document's whitelist and default egress-IP policy. The
// account setting is deliberately opt-in, independent of Excel/BPS routing.
var codexAllowedTimezones = map[string]bool{
	"Africa/Cairo": true, "Africa/Johannesburg": true, "America/Argentina/Buenos_Aires": true,
	"America/Chicago": true, "America/Denver": true, "America/Los_Angeles": true, "America/Mexico_City": true,
	"America/New_York": true, "America/Sao_Paulo": true, "America/Toronto": true, "America/Vancouver": true,
	"Asia/Bangkok": true, "Asia/Dubai": true, "Asia/Ho_Chi_Minh": true, "Asia/Jakarta": true, "Asia/Kolkata": true,
	"Asia/Kuala_Lumpur": true, "Asia/Manila": true, "Asia/Seoul": true, "Asia/Singapore": true, "Asia/Tokyo": true,
	"Australia/Perth": true, "Australia/Sydney": true, "Europe/Berlin": true, "Europe/Istanbul": true,
	"Europe/London": true, "Europe/Moscow": true, "Europe/Paris": true, "Pacific/Auckland": true, "Pacific/Honolulu": true,
}

var codexTimezoneTag = regexp.MustCompile(`<timezone>[^<]*</timezone>`)
var codexCurrentDateTag = regexp.MustCompile(`<current_date>\d{4}-\d{2}-\d{2}</current_date>`)

func (a *Account) IsCodexTimezoneRewriteEnabled() bool {
	if a == nil || a.Platform != PlatformOpenAI || a.Type != AccountTypeOAuth || a.IsShadow() || a.IsOpenAIAgentIdentity() || a.IsOpenAIPersonalAccessToken() {
		return false
	}
	for _, key := range []string{openAIAuthModeCredentialKey, openAIAuthModeLegacyCredentialKey} {
		mode := strings.ToLower(strings.TrimSpace(a.GetCredential(key)))
		if mode == "agentidentity" || mode == "agent_identity" {
			return false
		}
	}
	enabled, _ := a.Extra["openai_codex_timezone_rewrite"].(bool)
	return enabled
}

// Only inspect actual user input_text parts. Tool arguments, quoted examples,
// assistant text and self-closing unavailable dates are never rewritten.
func codexEnvironmentParts(body []byte, visit func(string, string)) {
	for i, item := range gjson.GetBytes(body, "input").Array() {
		if item.Get("role").String() != "user" || !item.Get("content").IsArray() {
			continue
		}
		for j, part := range item.Get("content").Array() {
			if part.Get("type").String() != "input_text" || part.Get("text").Type != gjson.String {
				continue
			}
			text := part.Get("text").String()
			if strings.HasPrefix(strings.TrimSpace(text), "<environment_context>") {
				visit("input."+strconv.Itoa(i)+".content."+strconv.Itoa(j)+".text", text)
			}
		}
	}
}

func rewriteCodexTimezone(body []byte, target string, now time.Time) ([]byte, bool, error) {
	if !json.Valid(body) {
		return body, false, errors.New("invalid timezone rewrite JSON")
	}
	location, err := time.LoadLocation(target)
	if err != nil {
		return body, false, err
	}
	date := now.In(location).Format("2006-01-02")
	out := body
	changed := false
	codexEnvironmentParts(body, func(path, text string) {
		if err != nil {
			return
		}
		end := strings.Index(text, "</environment_context>")
		if end < 0 {
			return
		}
		block := text[:end]
		replaceFirst := func(re *regexp.Regexp, replacement string) {
			if loc := re.FindStringIndex(block); loc != nil {
				block = block[:loc[0]] + replacement + block[loc[1]:]
			}
		}
		replaceFirst(codexTimezoneTag, "<timezone>"+target+"</timezone>")
		replaceFirst(codexCurrentDateTag, "<current_date>"+date+"</current_date>")
		next := block + text[end:]
		if next == text {
			return
		}
		var encoded bytes.Buffer
		encoder := json.NewEncoder(&encoded)
		encoder.SetEscapeHTML(false)
		if err = encoder.Encode(next); err != nil {
			return
		}
		// Surgical JSON replacement preserves every unrelated byte and unknown
		// field, including large integers and escaped WebSocket payloads.
		out, err = sjson.SetRawBytes(out, path, bytes.TrimSuffix(encoded.Bytes(), []byte{'\n'}))
		changed = true
	})
	if err != nil {
		return body, false, err
	}
	return out, changed, nil
}

type codexTimezoneCacheEntry struct {
	zone    string
	expires time.Time
	ready   chan struct{}
}

type codexTimezoneResolver struct {
	mu      sync.Mutex
	entries map[[32]byte]*codexTimezoneCacheEntry
	lookup  func(context.Context, string) (string, error)
	now     func() time.Time
}

func (r *codexTimezoneResolver) clock() time.Time {
	if r.now != nil {
		return r.now()
	}
	return time.Now()
}

func (r *codexTimezoneResolver) target(ctx context.Context, proxyURL string) string {
	key := sha256.Sum256([]byte(proxyURL)) // Never retain proxy credentials as cache keys.
	r.mu.Lock()
	if cached := r.entries[key]; cached != nil {
		if cached.ready != nil {
			ready := cached.ready
			r.mu.Unlock()
			select {
			case <-ready:
				return r.target(ctx, proxyURL)
			case <-ctx.Done():
				return codexTimezoneFallback
			}
		}
		if r.clock().Before(cached.expires) {
			zone := cached.zone
			r.mu.Unlock()
			return zone
		}
		delete(r.entries, key)
	}
	if r.entries == nil {
		r.entries = make(map[[32]byte]*codexTimezoneCacheEntry)
	}
	if len(r.entries) >= 1024 {
		// Prefer an expired/old completed entry; never evict an active lookup.
		var oldest [32]byte
		var expiry time.Time
		for k, v := range r.entries {
			if v.ready == nil && (expiry.IsZero() || v.expires.Before(expiry)) {
				oldest, expiry = k, v.expires
			}
		}
		if expiry.IsZero() {
			r.mu.Unlock()
			return codexTimezoneFallback
		}
		delete(r.entries, oldest)
	}
	entry := &codexTimezoneCacheEntry{ready: make(chan struct{})}
	r.entries[key] = entry
	r.mu.Unlock()
	lookup := r.lookup
	if lookup == nil {
		lookup = lookupCodexEgressTimezone
	}
	lookupCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	zone, err := lookup(lookupCtx, proxyURL)
	cancel()
	ttl := 24 * time.Hour
	if err != nil || zone == "" {
		ttl = 10 * time.Minute
		zone = codexTimezoneFallback
	}
	if !codexAllowedTimezones[zone] {
		zone = codexTimezoneFallback
	}
	r.mu.Lock()
	entry.zone, entry.expires = zone, r.clock().Add(ttl)
	ready := entry.ready
	entry.ready = nil
	close(ready)
	r.mu.Unlock()
	return zone
}

func lookupCodexEgressTimezone(ctx context.Context, proxyURL string) (string, error) {
	client, err := httpclient.GetClient(httpclient.Options{ProxyURL: proxyURL, Timeout: 3 * time.Second})
	if err != nil {
		return "", err
	}
	localClient := *client
	localClient.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://ipinfo.io/json", nil)
	if err != nil {
		return "", err
	}
	resp, err := localClient.Do(req) // No model request body or authentication headers are sent.
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", errors.New("timezone lookup unavailable")
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, (64<<10)+1))
	if err != nil || len(body) > 64<<10 {
		return "", errors.New("invalid timezone lookup response")
	}
	var result struct {
		Timezone string `json:"timezone"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", err
	}
	return strings.TrimSpace(result.Timezone), nil
}

func (s *OpenAIGatewayService) applyCodexTimezone(ctx context.Context, account *Account, body []byte) []byte {
	if !account.IsCodexTimezoneRewriteEnabled() {
		return body
	}
	matched := false
	codexEnvironmentParts(body, func(_ string, text string) {
		if end := strings.Index(text, "</environment_context>"); end >= 0 {
			matched = matched || codexTimezoneTag.MatchString(text[:end]) || codexCurrentDateTag.MatchString(text[:end])
		}
	})
	if !matched || !json.Valid(body) {
		return body
	}
	proxyURL := ""
	if account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}
	target := s.codexTimezone.target(ctx, proxyURL)
	out, _, err := rewriteCodexTimezone(body, target, s.codexTimezone.clock())
	if err != nil {
		return body
	} // An optional rewrite must not reject a business request.
	return out
}
