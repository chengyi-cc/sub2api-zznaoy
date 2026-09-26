package service

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	_ "time/tzdata"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

const timezoneTestEnvironment = `<environment_context><current_date>2020-01-01</current_date><timezone>Asia/Taipei</timezone></environment_context>`

func timezoneTestBody(t *testing.T, text string) []byte {
	t.Helper()
	encoded, err := json.Marshal(text)
	require.NoError(t, err)
	return []byte(`{"model":"gpt-5.6-sol","stream":false,"instructions":"test","input":[{"role":"user","content":[{"type":"input_text","text":` + string(encoded) + `}]}],"prompt_cache_key":"keep-session","unknown":9007199254740993}`)
}

func TestCodexTimezoneRewriteScopeAndBytePreservation(t *testing.T) {
	now := time.Date(2026, 9, 26, 17, 0, 0, 0, time.UTC)
	environment, err := json.Marshal(timezoneTestEnvironment)
	require.NoError(t, err)
	// Marshal escapes angle brackets, as real WebSocket payloads may do.
	require.Contains(t, string(environment), `\u003c`)
	body := []byte(`{
	"input": [
	 {"role":"user","content":[{"type":"input_text","text":` + string(environment) + `},{"type":"input_text","text":"Explain <timezone>Asia/Taipei</timezone>"}]},
	 {"role":"assistant","content":[{"type":"input_text","text":` + string(environment) + `}]},
	 {"type":"function_call_output","call_id":"keep","output":` + string(environment) + `},
	 {"role":"user","content":[{"type":"input_text","text":` + string(environment) + `}]}
	], "large": 9007199254740993, "unknown":{"raw":"\u003c"}, "prompt_cache_key":"unchanged"
}`)
	out, changed, err := rewriteCodexTimezone(body, "Asia/Singapore", now)
	require.NoError(t, err)
	require.True(t, changed)
	want := strings.ReplaceAll(timezoneTestEnvironment, "Asia/Taipei", "Asia/Singapore")
	want = strings.ReplaceAll(want, "2020-01-01", "2026-09-27")
	for _, index := range []string{"0", "3"} {
		require.Equal(t, want, gjson.GetBytes(out, "input."+index+".content.0.text").String())
	}
	for _, path := range []string{"input.0.content.1", "input.1", "input.2", "large", "unknown", "prompt_cache_key"} {
		require.Equal(t, gjson.GetBytes(body, path).Raw, gjson.GetBytes(out, path).Raw, path)
	}
	require.Contains(t, string(out), `], "large": 9007199254740993, "unknown":{"raw":"\u003c"}, "prompt_cache_key":"unchanged"`)
	again, changed, err := rewriteCodexTimezone(out, "Asia/Singapore", now)
	require.NoError(t, err)
	require.False(t, changed)
	require.Equal(t, out, again)
}

func TestCodexTimezoneRewriteDatesAndBoundaries(t *testing.T) {
	for _, tc := range []struct {
		name, text, zone, date string
		now                    time.Time
	}{
		{"previous day", timezoneTestEnvironment, "America/Los_Angeles", "2026-09-25", time.Date(2026, 9, 26, 1, 0, 0, 0, time.UTC)},
		{"DST next day", timezoneTestEnvironment, "Europe/Paris", "2026-07-02", time.Date(2026, 7, 1, 22, 30, 0, 0, time.UTC)},
		{"winter same day", timezoneTestEnvironment, "Europe/Paris", "2026-01-01", time.Date(2026, 1, 1, 22, 30, 0, 0, time.UTC)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, changed, err := rewriteCodexTimezone(timezoneTestBody(t, tc.text), tc.zone, tc.now)
			require.NoError(t, err)
			require.True(t, changed)
			require.Contains(t, gjson.GetBytes(out, "input.0.content.0.text").String(), "<current_date>"+tc.date+"</current_date>")
		})
	}
	t.Run("only first tag inside environment", func(t *testing.T) {
		text := `  <environment_context><current_date status="unavailable" /><timezone>old</timezone><timezone>second</timezone></environment_context><timezone>outside</timezone>`
		out, changed, err := rewriteCodexTimezone(timezoneTestBody(t, text), "Asia/Singapore", time.Now())
		require.NoError(t, err)
		require.True(t, changed)
		require.Equal(t, strings.Replace(text, "<timezone>old</timezone>", "<timezone>Asia/Singapore</timezone>", 1), gjson.GetBytes(out, "input.0.content.0.text").String())
	})
	for _, text := range []string{"Quoted " + timezoneTestEnvironment, "<environment_context><timezone>old</timezone>", `<environment_context><current_date status="unavailable" /></environment_context>`} {
		body := timezoneTestBody(t, text)
		out, changed, err := rewriteCodexTimezone(body, "Asia/Singapore", time.Now())
		require.NoError(t, err)
		require.False(t, changed)
		require.Equal(t, body, out)
	}
	for _, tc := range []struct {
		body []byte
		zone string
	}{{[]byte("{"), "Asia/Singapore"}, {timezoneTestBody(t, timezoneTestEnvironment), "Invalid/Zone"}} {
		out, changed, err := rewriteCodexTimezone(tc.body, tc.zone, time.Now())
		require.Error(t, err)
		require.False(t, changed)
		require.Equal(t, tc.body, out)
	}
}

func TestCodexTimezoneDisabledAndIneligibleAccountsAreUntouched(t *testing.T) {
	body := timezoneTestBody(t, timezoneTestEnvironment)
	for _, account := range []*Account{
		nil,
		{Platform: PlatformOpenAI, Type: AccountTypeOAuth},
		{Platform: PlatformOpenAI, Type: AccountTypeOAuth, Extra: map[string]any{"openai_codex_timezone_rewrite": false}},
		{Platform: PlatformOpenAI, Type: AccountTypeOAuth, Extra: map[string]any{"openai_codex_timezone_rewrite": "true"}},
		{Platform: PlatformOpenAI, Type: AccountTypeAPIKey, Extra: map[string]any{"openai_codex_timezone_rewrite": true}},
		{Platform: PlatformAnthropic, Type: AccountTypeOAuth, Extra: map[string]any{"openai_codex_timezone_rewrite": true}},
		{Platform: PlatformOpenAI, Type: AccountTypeOAuth, Credentials: map[string]any{"auth_mode": "agentIdentity"}, Extra: map[string]any{"openai_codex_timezone_rewrite": true}},
		{Platform: PlatformOpenAI, Type: AccountTypeOAuth, Credentials: map[string]any{"auth_mode": "personalAccessToken"}, Extra: map[string]any{"openai_codex_timezone_rewrite": true}},
	} {
		svc := &OpenAIGatewayService{}
		svc.codexTimezone.lookup = func(context.Context, string) (string, error) {
			t.Fatal("disabled/ineligible account must not perform lookup")
			return "", nil
		}
		require.False(t, account.IsCodexTimezoneRewriteEnabled())
		require.Equal(t, body, svc.applyCodexTimezone(context.Background(), account, body))
	}
	account := excelAccount()
	account.Extra["openai_codex_timezone_rewrite"] = true
	svc := &OpenAIGatewayService{}
	svc.codexTimezone.lookup = func(context.Context, string) (string, error) {
		t.Fatal("irrelevant or malformed input must not perform lookup")
		return "", nil
	}
	for _, body := range [][]byte{[]byte("{"), timezoneTestBody(t, "ordinary user message"), []byte(`{"input":[{"role":"user","content":[{"type":"input_text","text":"<environment_context><timezone>old</timezone></environment_context>"}]}],}`)} {
		require.Equal(t, body, svc.applyCodexTimezone(context.Background(), account, body))
	}
}

func TestCodexTimezoneResolverCaching(t *testing.T) {
	for _, tc := range []struct {
		name, result, want string
		err                error
		ttl                time.Duration
	}{
		{"positive", "America/Los_Angeles", "America/Los_Angeles", nil, 24 * time.Hour},
		{"outside whitelist", "Asia/Shanghai", codexTimezoneFallback, nil, 24 * time.Hour},
		{"empty", "", codexTimezoneFallback, nil, 10 * time.Minute},
		{"failed", "Europe/Paris", codexTimezoneFallback, errors.New("unavailable"), 10 * time.Minute},
	} {
		t.Run(tc.name, func(t *testing.T) {
			now := time.Date(2026, 9, 26, 0, 0, 0, 0, time.UTC)
			calls := 0
			proxyURL := "http://u:secret@127.0.0.1:1234"
			r := codexTimezoneResolver{now: func() time.Time { return now }, lookup: func(ctx context.Context, proxy string) (string, error) {
				calls++
				require.Equal(t, proxyURL, proxy)
				deadline, ok := ctx.Deadline()
				require.True(t, ok)
				require.InDelta(t, 3, time.Until(deadline).Seconds(), 0.5)
				return tc.result, tc.err
			}}
			require.Equal(t, tc.want, r.target(context.Background(), proxyURL))
			now = now.Add(tc.ttl - time.Second)
			require.Equal(t, tc.want, r.target(context.Background(), proxyURL))
			require.Equal(t, 1, calls)
			require.Len(t, r.entries, 1)
			require.Contains(t, r.entries, sha256.Sum256([]byte(proxyURL)))
			now = now.Add(time.Second)
			require.Equal(t, tc.want, r.target(context.Background(), proxyURL))
			require.Equal(t, 2, calls)
		})
	}
}

func TestCodexTimezoneResolverCoalescesConcurrentLookups(t *testing.T) {
	var calls atomic.Int32
	started, release := make(chan struct{}), make(chan struct{})
	r := codexTimezoneResolver{lookup: func(context.Context, string) (string, error) {
		calls.Add(1)
		close(started)
		<-release
		return "Europe/Paris", nil
	}}
	var wg sync.WaitGroup
	results := make(chan string, 20)
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); results <- r.target(context.Background(), "proxy") }()
	}
	<-started
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	require.Equal(t, codexTimezoneFallback, r.target(canceled, "proxy"))
	close(release)
	wg.Wait()
	close(results)
	for zone := range results {
		require.Equal(t, "Europe/Paris", zone)
	}
	require.Equal(t, int32(1), calls.Load())
}

func TestCodexTimezoneForwardPipelines(t *testing.T) {
	for _, mode := range []string{"native", "passthrough", "excel"} {
		for _, enabled := range []bool{false, true} {
			t.Run(mode+"/"+map[bool]string{false: "off", true: "on"}[enabled], func(t *testing.T) {
				account := excelAccount()
				account.Extra = map[string]any{"openai_codex_timezone_rewrite": enabled, "openai_excel_bps": mode == "excel", "openai_oauth_passthrough": mode == "passthrough"}
				proxyID := int64(12)
				account.ProxyID = &proxyID
				account.Proxy = &Proxy{ID: proxyID, Protocol: "socks5", Host: "127.0.0.1", Port: 4567}
				upstream := &httpUpstreamRecorder{err: errors.New("stop after capture")}
				svc := openAIClientToolsTestService(upstream)
				calls := 0
				svc.codexTimezone.lookup = func(_ context.Context, proxyURL string) (string, error) {
					calls++
					require.Equal(t, account.Proxy.URL(), proxyURL)
					return "Europe/Paris", nil
				}
				svc.codexTimezone.now = func() time.Time { return time.Date(2026, 9, 26, 23, 30, 0, 0, time.UTC) }
				body := timezoneTestBody(t, timezoneTestEnvironment)
				c, _ := gin.CreateTestContext(httptest.NewRecorder())
				c.Request = httptest.NewRequest("POST", "/v1/responses", bytes.NewReader(body))
				_, err := svc.Forward(context.Background(), c, account, body)
				require.Error(t, err)
				require.NotNil(t, upstream.lastReq)
				text := gjson.GetBytes(upstream.lastBody, `input.#(role=="user").content.0.text`).String()
				if enabled {
					require.Equal(t, 1, calls)
					require.Contains(t, text, "<timezone>Europe/Paris</timezone>")
					require.Contains(t, text, "<current_date>2026-09-27</current_date>")
				} else {
					require.Zero(t, calls)
					require.Equal(t, timezoneTestEnvironment, text)
				}
				require.Equal(t, int64(len(upstream.lastBody)), upstream.lastReq.ContentLength)
			})
		}
	}
}
