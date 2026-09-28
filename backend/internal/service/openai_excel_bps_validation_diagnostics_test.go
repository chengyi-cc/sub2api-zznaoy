package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestExcelBPSValidationParamAcrossResponseModes(t *testing.T) {
	for _, path := range []string{"/v1/responses", "/v1/responses/compact"} {
		for _, committed := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/committed=%t", path, committed), func(t *testing.T) {
				body, err := json.Marshal(gin.H{"model": "gpt-6-astra", "stream": committed, "input": []any{
					gin.H{"role": "user", "content": "earlier"},
					gin.H{"type": "agent_message", "author": "agent", "content": []any{gin.H{"type": "encrypted_content", "encrypted_content": "private-ciphertext"}}},
				}})
				require.NoError(t, err)
				rec := httptest.NewRecorder()
				c, _ := gin.CreateTestContext(rec)
				c.Request = httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
				if committed {
					stop := startOpenAISSEKeepalive(c, time.Hour)
					defer stop()
					value, exists := c.Get(openAICompactSSEKeepaliveKey)
					require.True(t, exists)
					require.True(t, value.(*openAICompactSSEKeepalive).beat())
				}
				up := &httpUpstreamRecorder{}
				svc := openAIClientToolsTestService(up)
				_, err = svc.Forward(context.Background(), c, excelAccount(), body)
				require.Error(t, err)
				require.Nil(t, up.lastReq, "invalid history must be rejected before sending")
				wire := rec.Body.String()
				require.NotContains(t, wire, "private-ciphertext")
				if committed {
					require.Equal(t, http.StatusOK, rec.Code)
					parts := strings.SplitN(wire, "data: ", 2)
					require.Len(t, parts, 2)
					require.Equal(t, "response.failed", gjson.Get(parts[1], "type").String())
					require.Equal(t, "input[1].content[0]", gjson.Get(parts[1], "response.error.param").String())
					require.NotEmpty(t, gjson.Get(parts[1], "response.id").String())
					require.True(t, gjson.Get(parts[1], "response.created_at").Exists())
				} else {
					require.Equal(t, http.StatusBadRequest, rec.Code)
					require.Equal(t, "input[1].content[0]", gjson.Get(wire, "error.param").String())
				}
			})
		}
	}
}

func TestExcelBPSTransportDiagnosticsRedactNetworkSecrets(t *testing.T) {
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	req, err := newExcelBPSRequest(context.Background(), []byte("private-body"), "private-token", "private-owner")
	require.NoError(t, err)
	trace := httptrace.ContextClientTrace(req.Context())
	require.Nil(t, req.GetBody, "model POST must not be implicitly rewound")
	payload, err := io.ReadAll(req.Body)
	require.NoError(t, err)
	require.Equal(t, "private-body", string(payload))
	require.NoError(t, req.Body.Close())
	trace.GetConn("private-host")
	trace.WroteHeaders()
	trace.WroteRequest(httptrace.WroteRequestInfo{})
	cause := &url.Error{Op: "Post", URL: "https://private-user:private-password@private-host/private-path", Err: io.ErrUnexpectedEOF}
	c.Set(OpsUpstreamStatusCodeKey, http.StatusServiceUnavailable)
	raw := recordExcelBPSTransportFailure(c, excelAccount(), req, cause)
	require.Zero(t, c.GetInt(OpsUpstreamStatusCodeKey), "connection failure is not an HTTP rejection")
	require.Equal(t, "unexpected_eof", gjson.GetBytes(raw, "error_kind").String())
	require.Equal(t, "awaiting_response_headers", gjson.GetBytes(raw, "transport.phase").String())
	require.True(t, gjson.GetBytes(raw, "transport.body_read").Bool())
	require.NotContains(t, string(raw), "private-")
	message, _ := c.Get(OpsUpstreamErrorMessageKey)
	require.NotContains(t, fmt.Sprint(message), "private-")
}
