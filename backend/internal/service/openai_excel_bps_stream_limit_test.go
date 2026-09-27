package service

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestExcelBPSConfiguredSSELimitCoversBothReaders(t *testing.T) {
	for _, tc := range []struct {
		name              string
		limit, size       int
		stream, wantError bool
	}{
		{"stream above old limit", 24 << 20, 17 << 20, true, false},
		{"nonstream above old limit", 24 << 20, 17 << 20, false, false},
		{"default gateway limit", 0, 17 << 20, true, false},
		{"configured bound still enforced", 1 << 20, 2 << 20, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			encrypted := strings.Repeat("x", tc.size)
			event, err := json.Marshal(map[string]any{
				"type": "response.completed", "response": map[string]any{
					"id": "resp_large", "status": "completed", "output": []any{map[string]any{"type": "compaction", "id": "cmp_large", "encrypted_content": encrypted}},
				},
			})
			require.NoError(t, err)
			upstream := &httpUpstreamRecorder{resp: excelBPSTestResponse("data: " + string(event) + "\n\n")}
			svc := openAIClientToolsTestService(upstream)
			svc.cfg.Gateway.MaxLineSize = tc.limit
			body, err := json.Marshal(map[string]any{"model": "gpt-5.6-sol", "input": "test", "stream": tc.stream})
			require.NoError(t, err)
			rec := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(rec)
			c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
			result, err := svc.Forward(context.Background(), c, excelAccount(), body)
			if tc.wantError {
				require.Error(t, err)
				require.Contains(t, rec.Body.String(), "basispoints_protocol_error")
				require.NotContains(t, rec.Body.String(), "response.completed")
				return
			}
			require.NoError(t, err)
			require.Equal(t, "response.completed", result.UpstreamTerminalEvent)
			payload := rec.Body.String()
			if tc.stream {
				for _, line := range strings.Split(payload, "\n") {
					if strings.HasPrefix(line, "data: ") {
						payload = gjson.Get(strings.TrimPrefix(line, "data: "), "response").Raw
					}
				}
			}
			actual := gjson.Get(payload, "output.0.encrypted_content").String()
			require.True(t, actual == encrypted, "large compaction must be preserved byte-for-byte")
		})
	}
}
