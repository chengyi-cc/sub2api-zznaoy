package service

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/service/basispoints"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func checkEncryptedHistoryNativeRoute(t *testing.T, body []byte) {
	t.Helper()
	require.Equal(t, "encrypted_message_history", basispoints.NativeFallbackReason(body))
	up := &httpUpstreamRecorder{resp: excelBPSTestResponse("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_native\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n")}
	svc := openAIClientToolsTestService(up)
	account := excelAccount()
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest("POST", "/v1/responses", bytes.NewReader(body))
	_, err := svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.Len(t, up.requests, 1)
	require.Equal(t, "chatgpt.com", up.lastReq.URL.Host)
	require.Equal(t, "encrypted_message_history", rec.Header().Get("X-Codex2API-Basispoints-Bypass"))
	sent := gjson.GetBytes(up.lastBody, "input").Array()
	original := gjson.GetBytes(body, "input").Array()
	found := 0
	for _, item := range original {
		encrypted := false
		for _, part := range item.Get("content").Array() {
			encrypted = encrypted || part.Get("type").String() == "encrypted_content"
		}
		if !encrypted {
			continue
		}
		var matching gjson.Result
		for _, candidate := range sent {
			if candidate.Get("id").String() == item.Get("id").String() {
				matching = candidate
				break
			}
		}
		want, err := json.Marshal(item.Value())
		require.NoError(t, err)
		got, err := json.Marshal(matching.Value())
		require.NoError(t, err)
		require.True(t, bytes.Equal(want, got), "encrypted message and attribution must survive unchanged")
		found++
	}
	require.Greater(t, found, 0)
	require.True(t, account.IsExcelBPSEnabled(), "request-only fallback must not disable the account")
}

func TestExcelBPSEncryptedHistoryUsesNativeWithoutLoss(t *testing.T) {
	for _, kind := range []string{"agent_message", "message"} {
		for _, stream := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s/stream=%t", kind, stream), func(t *testing.T) {
				body, err := json.Marshal(gin.H{"model": "gpt-6-astra", "stream": stream, "input": []any{gin.H{
					"type": kind, "id": "msg_opaque", "role": "user", "author": "/root/worker", "recipient": "/root",
					"content": []any{gin.H{"type": "input_text", "text": "Message Type: MESSAGE\nPayload:\n"}, gin.H{"type": "encrypted_content", "encrypted_content": "opaque-agent-result"}},
				}}})
				require.NoError(t, err)
				checkEncryptedHistoryNativeRoute(t, body)
			})
		}
	}
}

// Optional private fixtures stay outside the repository and never reach a network.
func TestExcelBPSCapturedEncryptedHistoryOffline(t *testing.T) {
	dir := os.Getenv("SUB2API_ENCRYPTED_HISTORY_CAPTURE_DIR")
	if dir == "" {
		t.Skip("private capture directory not supplied")
	}
	paths, err := filepath.Glob(filepath.Join(dir, "error-capture-*.json"))
	require.NoError(t, err)
	require.NotEmpty(t, paths)
	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) {
			raw, err := os.ReadFile(path)
			require.NoError(t, err)
			var capture map[string]json.RawMessage
			require.NoError(t, json.Unmarshal(raw, &capture))
			var encoded string
			require.NoError(t, json.Unmarshal(capture["request_wire_base64"], &encoded))
			body, err := base64.StdEncoding.DecodeString(encoded)
			require.NoError(t, err)
			checkEncryptedHistoryNativeRoute(t, body)
		})
	}
}
