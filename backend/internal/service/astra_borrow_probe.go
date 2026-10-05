package service

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/openai"
	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/google/uuid"
	"github.com/tidwall/gjson"
)

// Only non-secret evidence is exposed; raw tickets and cookies stay local.
type AstraBorrowProbeDetails struct {
	MintStatus           int  `json:"mint_status"`
	ContinueStatus       int  `json:"continue_status"`
	TicketLength         int  `json:"ticket_length"`
	ContinueTicketLength int  `json:"continue_ticket_length"`
	NewTicket            bool `json:"new_ticket"`
}

// Match the reference harvest identity, including its Astra-specific version floor.
func applyAstraBorrowProbeIdentity(h http.Header) {
	if h == nil {
		return
	}
	ensureCodexIdentityHeaders(h)
	enforceCodexIdentityHeaders(h)
	if CompareVersions(h.Get("version"), "0.153.4") < 0 {
		h.Set("version", "0.153.4")
		h.Set("user-agent", buildCodexCLIUserAgent("0.153.4"))
		h.Set("originator", openai.CodexDefaultOriginator)
	}
}

type astraBorrowShot struct {
	status     int
	state      string
	cookies    []*http.Cookie
	receivedAt time.Time
}

type astraBorrowOwnedSlotKey struct{}

func (s *AstraBorrowService) fire(ctx context.Context, account *Account, headers http.Header, profile *tlsfingerprint.Profile, proxy, state, cookie string) (astraBorrowShot, error) {
	var out astraBorrowShot
	ctx, cancel := context.WithTimeout(ctx, 45*time.Second)
	defer cancel()
	// Probe calls use the same account admission counters, but never recurse through
	// the borrowing layer or run business retry/failover logic.
	ownedSlot, _ := ctx.Value(astraBorrowOwnedSlotKey{}).(int64)
	if s.gateway.concurrencyService != nil && ownedSlot != account.ID {
		slot, err := s.gateway.concurrencyService.AcquireAccountSlot(ctx, account.ID, account.Concurrency)
		if err != nil || slot == nil || !slot.Acquired {
			return out, errors.New("astra_account_busy")
		}
		defer slot.ReleaseFunc()
	}
	if err := s.gateway.acquireOpenAIRPMForSend(ctx, account); err != nil {
		return out, errors.New("astra_rate_limited")
	}
	body := []byte(`{"model":"gpt-6-astra","instructions":"Reply with OK.","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Reply with OK."}]}],"stream":true,"store":false,"parallel_tool_calls":true,"include":["reasoning.encrypted_content"]}`)
	// Match the upstream two-shot probe: full Responses over fresh HTTP/1.1 connections.
	// req.Close alone does not prevent HTTP/2 connection reuse.
	ctx = WithHTTPUpstreamRedirectsDisabled(WithHTTPUpstreamProfile(ctx, HTTPUpstreamProfileOpenAIHarvest))
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, chatgptCodexURL, bytes.NewReader(body))
	if err != nil {
		return out, errors.New("astra_probe_request_failed")
	}
	req.Header = headers.Clone()
	req.Host = "chatgpt.com"
	req.Close = true
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("OpenAI-Beta", "responses=experimental")
	for _, key := range []string{"Cookie", "x-codex-turn-state", "Content-Length", "Content-Encoding", "conversation_id", "x-codex-turn-metadata", responsesLiteHeaderKey} {
		req.Header.Del(key)
	}
	req.Header.Set("session_id", uuid.NewString())
	if state != "" {
		req.Header.Set("x-codex-turn-state", state)
	}
	if cookie != "" {
		req.Header.Set("Cookie", cookie)
	}
	req.GetBody = nil
	resp, err := s.gateway.httpUpstream.DoWithTLS(req, proxy, account.ID, account.Concurrency, profile)
	if err != nil {
		if resp != nil && resp.Body != nil {
			_ = resp.Body.Close()
		}
		if ctx.Err() != nil {
			return out, errors.New("astra_probe_timeout")
		}
		return out, errors.New("astra_network_error")
	}
	if resp == nil || resp.Body == nil {
		return out, errors.New("astra_stream_incomplete")
	}
	defer func() { _ = resp.Body.Close() }()
	out.status = resp.StatusCode
	if resp.StatusCode != http.StatusOK {
		return out, fmt.Errorf("astra_upstream_%d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "" && !strings.HasPrefix(strings.ToLower(ct), "text/event-stream") {
		return out, errors.New("astra_stream_invalid")
	}
	out.state = strings.TrimSpace(resp.Header.Get("x-codex-turn-state"))
	if len(out.state) > 16384 {
		return out, errors.New("astra_ticket_invalid")
	}
	out.cookies = resp.Cookies()
	out.receivedAt = time.Now()
	data, err := io.ReadAll(io.LimitReader(resp.Body, (1<<20)+1))
	if err != nil || len(data) > 1<<20 {
		return out, errors.New("astra_stream_incomplete")
	}
	if err = validateAstraBorrowStream(data); err != nil {
		return out, err
	}
	if ctx.Err() != nil {
		return out, errors.New("astra_probe_timeout")
	}
	return out, nil
}

// Require a genuine completed terminal with text and the requested model. A 200,
// created event, [DONE], partial text, failed/incomplete terminal or wrong model
// cannot qualify a source or target. Parse full SSE events, including multiline data.
func validateAstraBorrowStream(data []byte) error {
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 4096), 1<<20)
	var lines []string
	completed, text := false, false
	flush := func() error {
		if len(lines) == 0 {
			return nil
		}
		payload := strings.Join(lines, "\n")
		lines = nil
		if payload == "[DONE]" {
			if !completed {
				return errors.New("astra_stream_incomplete")
			}
			return nil
		}
		if !json.Valid([]byte(payload)) {
			return errors.New("astra_stream_invalid")
		}
		e := gjson.Parse(payload)
		if model := e.Get("response.model").String(); model != "" && model != astraBorrowModel {
			return errors.New("astra_model_mismatch")
		}
		switch e.Get("type").String() {
		case "error", "response.failed", "response.incomplete", "response.cancelled", "response.canceled":
			return errors.New("astra_stream_failed")
		case "response.output_text.delta":
			text = text || strings.TrimSpace(e.Get("delta").String()) != ""
		case "response.completed", "response.done":
			if completed || e.Get("response.status").String() != "completed" || e.Get("response.model").String() != astraBorrowModel || (e.Get("response.error").Exists() && e.Get("response.error").Type != gjson.Null) {
				return errors.New("astra_stream_failed")
			}
			for _, item := range e.Get("response.output").Array() {
				for _, content := range item.Get("content").Array() {
					if content.Get("type").String() == "output_text" && strings.TrimSpace(content.Get("text").String()) != "" {
						text = true
					}
				}
			}
			completed = true
		}
		return nil
	}
	for scanner.Scan() {
		line := strings.TrimSuffix(scanner.Text(), "\r")
		if line == "" {
			if err := flush(); err != nil {
				return err
			}
		} else if strings.HasPrefix(line, "data:") {
			lines = append(lines, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
		}
	}
	if scanner.Err() != nil {
		return errors.New("astra_stream_invalid")
	}
	if err := flush(); err != nil {
		return err
	}
	if !completed || !text {
		return errors.New("astra_stream_incomplete")
	}
	return nil
}

func (s *AstraBorrowService) probeTarget(ctx context.Context, target *Account, headers http.Header, profile *tlsfingerprint.Profile, route astraBorrowRoute, proxy string) (details AstraBorrowProbeDetails, resultErr error) {
	seed := (&http.Cookie{Name: "__oailb", Value: route.cookie.Value}).String()
	first, err := s.fire(ctx, target, headers, profile, proxy, "", seed)
	details.MintStatus, details.TicketLength = first.status, len(first.state)
	if err != nil {
		return details, err
	}
	if astraCookiesChanged(first.cookies, route.cookie.Value) {
		return details, errors.New("astra_route_changed")
	}
	if first.state == "" {
		return details, errors.New("astra_ticket_missing")
	}
	// __cflb belongs to the target response, never to the source account.
	cookies := seed
	for _, c := range first.cookies {
		if c.Name == "__cflb" && c.Value != "" && c.MaxAge >= 0 && (c.Expires.IsZero() || time.Now().Before(c.Expires)) {
			cookies += "; " + (&http.Cookie{Name: c.Name, Value: c.Value}).String()
			break
		}
	}
	if !time.Now().Before(route.expires) {
		return details, errors.New("astra_route_expired")
	}
	second, err := s.fire(ctx, target, headers, profile, proxy, first.state, cookies)
	details.ContinueStatus, details.ContinueTicketLength = second.status, len(second.state)
	details.NewTicket = second.state != "" && second.state != first.state
	if err != nil {
		return details, err
	}
	if astraCookiesChanged(second.cookies, route.cookie.Value) {
		return details, errors.New("astra_route_changed")
	}
	if second.state != "" && second.state != first.state {
		return details, errors.New("astra_ticket_changed")
	}
	return details, nil
}

func astraRouteChanged(resp *http.Response, value string) bool {
	return resp != nil && astraCookiesChanged(resp.Cookies(), value)
}
func astraCookiesChanged(cookies []*http.Cookie, value string) bool {
	for _, c := range cookies {
		if c.Name == "__oailb" && (c.Value != value || c.MaxAge < 0 || (!c.Expires.IsZero() && !time.Now().Before(c.Expires))) {
			return true
		}
	}
	return false
}

func astraBorrowCookie(cookies []*http.Cookie, now time.Time, ttl int) (*http.Cookie, time.Time) {
	var result *http.Cookie
	var expires time.Time
	for _, c := range cookies {
		if c.Name != "__oailb" || !c.Secure || (c.Domain != "" && strings.TrimPrefix(c.Domain, ".") != "chatgpt.com") {
			continue
		}
		scope := c.Path
		if scope == "" {
			scope = "/backend-api/codex"
		}
		path := "/backend-api/codex/responses"
		if path != scope && !(strings.HasPrefix(path, scope) && (strings.HasSuffix(scope, "/") || strings.HasPrefix(strings.TrimPrefix(path, scope), "/"))) {
			continue
		}
		if c.Value == "" || c.MaxAge < 0 || (c.MaxAge == 0 && !c.Expires.IsZero() && !now.Before(c.Expires)) {
			return nil, time.Time{}
		}
		if c.MaxAge == 0 && c.Expires.IsZero() {
			continue
		}
		expires = now.Add(time.Duration(ttl) * time.Second)
		if c.MaxAge > 0 && c.MaxAge < ttl {
			expires = now.Add(time.Duration(c.MaxAge) * time.Second)
		} else if c.MaxAge == 0 && c.Expires.Before(expires) {
			expires = c.Expires
		}
		copy := *c
		result = &copy
	}
	return result, expires
}

func astraSetRouteCookie(headers http.Header, value string) {
	var kept []string
	for _, header := range headers.Values("Cookie") {
		for _, part := range strings.Split(header, ";") {
			part = strings.TrimSpace(part)
			name, _, _ := strings.Cut(part, "=")
			if part != "" && strings.TrimSpace(name) != "__oailb" {
				kept = append(kept, part)
			}
		}
	}
	kept = append(kept, (&http.Cookie{Name: "__oailb", Value: value}).String())
	headers.Set("Cookie", strings.Join(kept, "; "))
}
