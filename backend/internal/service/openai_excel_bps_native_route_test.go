package service

import (
	"context"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"net/http/httptest"
	"testing"
)

func TestExcelBPSNativeCapabilityDispatchBeforeSend(t *testing.T) {
	for _, body := range []string{
		`{"model":"gpt-6-astra","stream":true,"input":"hello","tools":[{"type":"web_search"}]}`,
		`{"model":"gpt-6-astra","stream":true,"input":"hello","tool_choice":{"type":"function","name":"echo"},"tools":[{"type":"function","name":"echo"}]}`,
	} {
		upstream := &httpUpstreamRecorder{resp: excelBPSTestResponse("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_native\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n")}
		svc := openAIClientToolsTestService(upstream)
		account := excelAccount()
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		c.Request = httptest.NewRequest("POST", "/v1/responses", nil)
		_, err := svc.Forward(context.Background(), c, account, []byte(body))
		require.NoError(t, err)
		require.Equal(t, "chatgpt.com", upstream.lastReq.URL.Host)
		require.Len(t, upstream.requests, 1)
		require.Equal(t, gjson.Get(body, "tools").Raw, gjson.GetBytes(upstream.lastBody, "tools").Raw)
		require.True(t, account.IsExcelBPSEnabled())
	}
}
