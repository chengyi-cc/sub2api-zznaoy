package service

import (
	"context"
	"errors"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestExcelBPSFlushFailureStopsDeliveryButRetainsUsage(t *testing.T) {
	w := newOpenAIResponseFlushRecorder()
	w.flushError = io.ErrClosedPipe
	wire := "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n" +
		"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_usage\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":7,\"output_tokens\":5}}}\n\n"
	u := &httpUpstreamRecorder{resp: &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(wire))}}
	svc := openAIClientToolsTestService(u)
	c, _ := gin.CreateTestContext(w)
	body := []byte("{\"model\":\"gpt-6-astra\",\"stream\":true,\"input\":\"test\"}")
	c.Request = httptest.NewRequest("POST", "/v1/responses", strings.NewReader(string(body)))
	r, err := svc.Forward(context.Background(), c, excelAccount(), body)
	require.NoError(t, err)
	require.True(t, r.ClientDisconnect)
	require.Equal(t, 1, w.flushErrorCalls)
	require.Equal(t, 7, r.Usage.InputTokens)
	require.Equal(t, 5, r.Usage.OutputTokens)
	out, _ := w.snapshot()
	require.NotContains(t, out, "response.completed")
}

func TestExcelBPSConfiguredIdleTimeoutReturnsFailureWithoutReplay(t *testing.T) {
	r, w := io.Pipe()
	defer w.Close()
	calls := 0
	u := &bpsCancellationUpstream{send: func(*http.Request, string) (*http.Response, error) {
		calls++
		return &http.Response{StatusCode: 200, Header: http.Header{}, Body: r}, nil
	}}
	svc := openAIClientToolsTestService(nil)
	svc.httpUpstream = u
	svc.cfg.Gateway.StreamDataIntervalTimeout = 1
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	body := []byte("{\"model\":\"gpt-6-astra\",\"stream\":true,\"input\":\"test\"}")
	c.Request = httptest.NewRequest("POST", "/v1/responses", strings.NewReader(string(body))).WithContext(ctx)
	go func() {
		_, _ = w.Write([]byte("data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_idle\",\"output\":[]}}\n\n"))
	}()
	result, err := svc.Forward(ctx, c, excelAccount(), body)
	require.True(t, errors.Is(err, errOpenAISSEIdle))
	require.NotNil(t, result)
	require.False(t, result.ClientDisconnect)
	require.Equal(t, 1, calls)
	require.Contains(t, rec.Body.String(), "basispoints_stream_timeout")
	require.NotContains(t, rec.Body.String(), "response.completed")
}
