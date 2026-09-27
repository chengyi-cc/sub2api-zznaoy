package service

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestExcelBPSCatalogGatewayIsolation(t *testing.T) {
	for _, mode := range []string{"same thread", "other thread", "other key", "other owner", "other account", "session only", "anonymous", "explicit empty"} {
		t.Run(mode, func(t *testing.T) {
			upstream := &httpUpstreamRecorder{}
			svc := openAIClientToolsTestService(upstream)
			account := excelAccount()
			thread := t.Name()
			keyID := int64(431)
			if mode == "session only" {
				thread = ""
			}
			if mode == "anonymous" {
				keyID = 0
			}
			request := map[string]any{"model": "gpt-5.6-sol", "input": "hello", "tools": []any{map[string]any{
				"type": "function", "name": "echo_cache_gate", "parameters": map[string]any{"type": "object"},
			}}}
			post := func() []byte {
				t.Helper()
				body, err := json.Marshal(request)
				require.NoError(t, err)
				upstream.responses = []*http.Response{excelBPSTestResponse("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_catalog\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n")}
				c, _ := gin.CreateTestContext(httptest.NewRecorder())
				c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
				c.Request.Header.Set("session_id", t.Name())
				c.Request.Header.Set("thread-id", thread)
				if keyID > 0 {
					c.Set("api_key", &APIKey{ID: keyID})
				}
				_, err = svc.Forward(context.Background(), c, account, body)
				require.NoError(t, err)
				return append([]byte(nil), upstream.lastBody...)
			}
			require.Contains(t, string(post()), "echo_cache_gate")
			delete(request, "tools")
			switch mode {
			case "other thread":
				thread += "-sibling"
			case "other key":
				keyID++
			case "other owner":
				account.Credentials["chatgpt_account_id"] = "other-owner"
			case "other account":
				account.ID++
			case "explicit empty":
				request["tools"] = []any{}
			}
			result := string(post())
			if mode == "same thread" {
				require.Contains(t, result, "echo_cache_gate")
			} else {
				require.NotContains(t, result, "echo_cache_gate")
			}
		})
	}
}

func TestExcelBPSDisabledHostedToolsStayOnBPS(t *testing.T) {
	upstream := &httpUpstreamRecorder{resp: excelBPSTestResponse("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_none\",\"status\":\"completed\",\"output\":[]}}\n\n")}
	svc := openAIClientToolsTestService(upstream)
	body, err := json.Marshal(map[string]any{"model": "gpt-5.6-sol", "input": "hello", "tool_choice": "none", "tools": []any{map[string]any{"type": "web_search"}}})
	require.NoError(t, err)
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	_, err = svc.Forward(context.Background(), c, excelAccount(), body)
	require.NoError(t, err)
	require.Equal(t, "/basispoints/api/responses", upstream.lastReq.URL.Path)
	require.Empty(t, recorder.Header().Get("X-Codex2API-Basispoints-Bypass"))
}
