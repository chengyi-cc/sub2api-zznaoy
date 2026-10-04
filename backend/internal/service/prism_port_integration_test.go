package service

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestPrismCompatibilityErrorAfterHeartbeatKeepsValidSSE(t *testing.T) {
	_, account := prismTestService("")
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("POST", "/v1/messages", nil)
	c.Header("Content-Type", "text/event-stream")
	_, _ = c.Writer.WriteString(": keepalive\n\n")
	c.Writer.Flush()
	require.Error(t, rejectPrismCompatibility(c, account, []byte(`{"model":"gpt-6.1-sol"}`), ""))
	require.Contains(t, w.Body.String(), "event: error\ndata: ")
	for _, line := range strings.Split(w.Body.String(), "\n") {
		if strings.HasPrefix(line, "data: ") {
			require.True(t, json.Valid([]byte(strings.TrimPrefix(line, "data: "))))
		}
	}
}

func TestPrismCompatibilityEndpointsCannotBypassSelectedRoute(t *testing.T) {
	s, account := prismTestService("http://127.0.0.1:1")
	for _, endpoint := range []string{"/v1/chat/completions", "/v1/messages"} {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest("POST", endpoint, nil)
		body := []byte(`{"model":"gpt-6.1-sol","messages":[{"role":"user","content":"test"}]}`)
		var err error
		if endpoint == "/v1/messages" {
			_, err = s.ForwardAsAnthropic(context.Background(), c, account, body, "", "")
		} else {
			_, err = s.ForwardAsChatCompletions(context.Background(), c, account, body, "", "")
		}
		require.ErrorContains(t, err, "requires HTTP /v1/responses")
		require.Equal(t, 400, w.Code)
		require.True(t, IsPrismBrowserAttempt(c, account.ID))
	}
}

func TestPrismWebSocketGuardPreservesNativeModelsAndRejectsSwitch(t *testing.T) {
	_, account := prismTestService("")
	account.Extra[PrismBrowserModelsKey] = []string{"gpt-6.1-sol"}
	called := 0
	hooks := &OpenAIWSIngressHooks{MapRequestModel: func(_ int, model string) (string, error) {
		called++
		if model == "alias" {
			return "gpt-6.1-sol", nil
		}
		return model, nil
	}}
	guarded := withPrismBrowserModelGuard(account, hooks)
	model, err := guarded.MapRequestModel(1, "gpt-6-astra")
	require.NoError(t, err)
	require.Equal(t, "gpt-6-astra", model)
	_, err = guarded.MapRequestModel(2, "alias")
	require.ErrorContains(t, err, "Prism")
	require.Equal(t, 2, called)
	_, err = hooks.MapRequestModel(2, "alias")
	require.NoError(t, err, "original hooks must not be mutated")
}

func TestPrismCannotEnableNonSubscriptionCredentials(t *testing.T) {
	_, a := prismTestService("")
	for _, mode := range []string{"agent_identity", "personal_access_token"} {
		a.Credentials["auth_mode"] = mode
		require.False(t, accountHasPrismBrowser(a))
	}
}
