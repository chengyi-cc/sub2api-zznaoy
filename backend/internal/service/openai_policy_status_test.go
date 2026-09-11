package service

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestOpenAIPolicyStatus_Forward(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, policy := range []struct {
		code   string
		status int
	}{
		{"session_blocked_by_cyber_policy", 403}, {"cyber_policy", 400},
	} {
		for _, pool := range []bool{false, true} {
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/pool=%v/stream=%v", policy.code, pool, stream), func(t *testing.T) {
					body := fmt.Sprintf(`{"error":{"code":%q,"type":"permission_error","message":"Blocked by security policy","internal_token":"sk-private"},"debug":"private"}`, policy.code)
					upstream := &httpUpstreamRecorder{resp: &http.Response{
						StatusCode: policy.status,
						Header:     http.Header{"Content-Type": {"application/json"}, "Retry-After": {"17"}, "Set-Cookie": {"secret=value"}},
						Body:       io.NopCloser(strings.NewReader(body)),
					}}
					svc := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream}
					account := &Account{
						ID: 124, Platform: PlatformOpenAI, Type: AccountTypeAPIKey,
						Concurrency: 1, Status: StatusActive, Schedulable: true,
						Credentials: map[string]any{"api_key": "sk-test", "base_url": "https://upstream.example", "pool_mode": pool, "pool_mode_retry_status_codes": []any{400, 403}},
						Extra:       map[string]any{"openai_passthrough": true},
					}
					rec := httptest.NewRecorder()
					c, _ := gin.CreateTestContext(rec)
					c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
					requestBody := []byte(fmt.Sprintf(`{"model":"gpt-5.2","stream":%v,"input":"hello"}`, stream))
					_, err := svc.Forward(context.Background(), c, account, requestBody)
					require.Error(t, err)
					var failover *UpstreamFailoverError
					require.False(t, errors.As(err, &failover))
					require.Len(t, upstream.requests, 1)
					require.Equal(t, policy.status, rec.Code)
					require.Equal(t, policy.code, gjson.Get(rec.Body.String(), "error.code").String())
					require.Equal(t, "permission_error", gjson.Get(rec.Body.String(), "error.type").String())
					require.Equal(t, "Blocked by security policy", gjson.Get(rec.Body.String(), "error.message").String())
					require.NotContains(t, rec.Body.String(), "private")
					require.Empty(t, rec.Header().Get("Retry-After"))
					require.Empty(t, rec.Header().Get("Set-Cookie"))
					require.Equal(t, "no-store", rec.Header().Get("Cache-Control"))
					require.NotNil(t, GetOpsCyberPolicy(c))
					require.Equal(t, policy.status, GetOpsCyberPolicy(c).UpstreamStatus)
					events := c.MustGet(OpsUpstreamErrorsKey).([]*OpsUpstreamErrorEvent)
					require.Equal(t, policy.status, events[len(events)-1].UpstreamStatusCode)
					require.Equal(t, "http_error", events[len(events)-1].Kind)
					require.False(t, svc.shouldFailoverOpenAIUpstreamResponse(account, policy.status, "", []byte(body)))
				})
			}
		}
	}
}

func TestOpenAIPolicyStatus_Detection(t *testing.T) {
	for _, payload := range []string{
		`{"error":{"code":"session_blocked_by_cyber_policy","message":"blocked"}}`,
		`{"response":{"error":{"code":" SESSION_BLOCKED_BY_CYBER_POLICY ","message":"blocked"}}}`,
	} {
		hit, code, message := detectOpenAICyberPolicy([]byte(payload))
		require.True(t, hit)
		require.Equal(t, "session_blocked_by_cyber_policy", code)
		require.Equal(t, "blocked", message)
	}
	for _, code := range []string{"not_cyber_policy", "cyber_policy_suffix", "session_blocked_by_cyber_policy_other"} {
		hit, _, _ := detectOpenAICyberPolicy([]byte(fmt.Sprintf(`{"error":{"code":%q}}`, code)))
		require.False(t, hit)
	}
}
