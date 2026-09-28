package service

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/ctxkey"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/service/basispoints"
	"github.com/Wei-Shaw/sub2api/internal/util/transportdiag"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"go.uber.org/zap"
	"go.uber.org/zap/zaptest/observer"
)

func TestExcelBPSTimingSeparatesCompactionBufferAndRepair(t *testing.T) {
	base := time.Now()
	ctx := context.WithValue(context.Background(), ctxkey.RequestStartTime, base)
	timing := newExcelBPSRequestTiming(ctx, base.Add(time.Second), true)
	for _, point := range []struct {
		stage basispoints.StreamStage
		ms    int
	}{
		{basispoints.StreamToolReady, 2000}, {basispoints.StreamMessageCompleted, 3000},
		{basispoints.StreamCompactionStarted, 4000}, {basispoints.StreamCompactionCompleted, 79000},
		{basispoints.StreamUpstreamCompleted, 80000}, {basispoints.StreamValidationStarted, 80000},
		{basispoints.StreamRepairStarted, 81000}, {basispoints.StreamRepairCompleted, 84000},
		{basispoints.StreamValidationCompleted, 85000}, {basispoints.StreamToolEmitted, 86000},
	} {
		timing.markAt(string(point.stage), base.Add(time.Duration(point.ms)*time.Millisecond))
	}
	values := timing.snapshot(base.Add(87 * time.Second))
	require.EqualValues(t, 75000, values["compaction_duration_ms"])
	require.EqualValues(t, 3000, values["repair_duration_ms"])
	require.EqualValues(t, 5000, values["validation_duration_ms"])
	require.EqualValues(t, 1, values["compactions_completed"])
	require.EqualValues(t, 84000, values["first_tool_buffer_ms"])
	require.EqualValues(t, 77000, values["last_message_to_completion_ms"])
	require.EqualValues(t, 1, values["tool_repairs"])
	require.NotContains(t, values, "first_headers_ms")
	require.NotContains(t, values, "attachment_duration_ms")
}

func TestExcelBPSTimingPreservesTransportTraceAndOmitsSecrets(t *testing.T) {
	timing := newExcelBPSRequestTiming(context.Background(), time.Now(), true)
	req, err := newExcelBPSRequest(context.Background(), []byte(`{"private":"sensitive-body"}`), "sensitive-token", "sensitive-account")
	require.NoError(t, err)
	existing := transportdiag.FromContext(req.Context())
	req, finish := timing.beginHTTP(req)
	require.Same(t, existing, transportdiag.FromContext(req.Context()))
	trace := httptrace.ContextClientTrace(req.Context())
	trace.GetConn("private-host:443")
	trace.DNSStart(httptrace.DNSStartInfo{Host: "private-host"})
	trace.DNSDone(httptrace.DNSDoneInfo{})
	trace.GotConn(httptrace.GotConnInfo{Reused: true})
	trace.WroteRequest(httptrace.WroteRequestInfo{})
	trace.GotFirstResponseByte()
	finish(&http.Response{StatusCode: 200}, nil)
	require.Equal(t, true, existing.Snapshot()["request_written"])
	values := timing.snapshot(time.Now())
	require.EqualValues(t, 1, values["http_attempts"])
	attempts := values["http_transport"].([]map[string]any)
	require.Equal(t, true, attempts[0]["connection_reused"])
	require.Contains(t, attempts[0], "request_written_ms")
	raw, err := json.Marshal(values)
	require.NoError(t, err)
	for _, secret := range []string{"sensitive-body", "sensitive-token", "sensitive-account", "private-host"} {
		require.NotContains(t, string(raw), secret)
	}
}

func TestExcelBPSTimingDisabledAndConcurrentCancellationSnapshot(t *testing.T) {
	ctx := context.Background()
	disabled := newExcelBPSRequestTiming(ctx, time.Now(), false)
	require.Nil(t, disabled)
	disabled.observe(basispoints.StreamFirstEvent)
	req := httptest.NewRequest("POST", "https://example.invalid", nil)
	same, finish := disabled.beginHTTP(req)
	require.Same(t, req, same)
	finish(nil, context.Canceled)
	disabled.finish(ctx, 1, nil, context.Canceled)
	timing := newExcelBPSRequestTiming(ctx, time.Now(), true)
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				timing.observe(basispoints.StreamFirstEvent)
				timing.snapshot(time.Now())
			}
		}()
	}
	wg.Wait()
}

func TestExcelBPSForwardCompactionConfigAndTiming(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, tc := range []struct {
		name      string
		threshold int
		enabled   bool
	}{{"disabled-diagnostics", 0, false}, {"enabled-diagnostics", 320000, true}} {
		t.Run(tc.name, func(t *testing.T) {
			wire := "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_excel\",\"status\":\"completed\",\"output\":[]}}\n\n"
			upstream := &httpUpstreamRecorder{resp: &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(wire))}}
			svc := openAIClientToolsTestService(upstream)
			svc.cfg.Gateway.ExcelBPS.CompactionThresholdTokens = &tc.threshold
			svc.cfg.Gateway.ExcelBPS.LogRequestTiming = tc.enabled
			body := []byte(`{"model":"gpt-5.6-sol","stream":true,"input":"secret-prompt"}`)
			rec := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(rec)
			core, logs := observer.New(zap.InfoLevel)
			ctx := logger.IntoContext(context.Background(), zap.New(core).With(zap.String("request_id", "request-test")))
			c.Request = httptest.NewRequest("POST", "/v1/responses", bytes.NewReader(body)).WithContext(ctx)
			_, err := svc.Forward(ctx, c, excelAccount(), body)
			require.NoError(t, err)
			if tc.threshold == 0 {
				require.Equal(t, "[]", gjson.GetBytes(upstream.lastBody, "context_management").Raw)
			} else {
				require.EqualValues(t, tc.threshold, gjson.GetBytes(upstream.lastBody, "context_management.0.compact_threshold").Int())
			}
			entries := logs.FilterMessage("excel_bps_request_timing").All()
			if !tc.enabled {
				require.Empty(t, entries)
				return
			}
			require.Len(t, entries, 1)
			fields := entries[0].ContextMap()
			require.Equal(t, "request-test", fields["request_id"])
			raw, err := json.Marshal(fields)
			require.NoError(t, err)
			require.True(t, gjson.GetBytes(raw, "timing.succeeded").Bool())
			require.EqualValues(t, 1, gjson.GetBytes(raw, "timing.http_attempts").Int())
			require.EqualValues(t, tc.threshold, gjson.GetBytes(raw, "timing.compaction_thresholds.0").Int())
			require.NotContains(t, string(raw), "secret-prompt")
			require.NotContains(t, string(raw), "test-token")
		})
	}
}

func TestExcelBPSTimingLogsFailureWithoutInventingUnreachedStages(t *testing.T) {
	gin.SetMode(gin.TestMode)
	upstream := &httpUpstreamRecorder{resp: &http.Response{StatusCode: 403, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(`{"error":{"message":"private-upstream-body"}}`))}}
	svc := openAIClientToolsTestService(upstream)
	svc.cfg.Gateway.ExcelBPS.LogRequestTiming = true
	body := []byte(`{"model":"gpt-5.6-sol","input":"private-prompt"}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	core, logs := observer.New(zap.InfoLevel)
	ctx := logger.IntoContext(context.Background(), zap.New(core))
	c.Request = httptest.NewRequest("POST", "/v1/responses", bytes.NewReader(body)).WithContext(ctx)
	_, err := svc.Forward(ctx, c, excelAccount(), body)
	require.Error(t, err)
	entries := logs.FilterMessage("excel_bps_request_timing").All()
	require.Len(t, entries, 1)
	raw, err := json.Marshal(entries[0].ContextMap())
	require.NoError(t, err)
	require.False(t, gjson.GetBytes(raw, "timing.succeeded").Bool())
	require.EqualValues(t, 403, gjson.GetBytes(raw, "timing.last_http_status").Int())
	require.False(t, gjson.GetBytes(raw, "timing.first_event_ms").Exists())
	for _, secret := range []string{"private-upstream-body", "private-prompt", "test-token"} {
		require.NotContains(t, string(raw), secret)
	}
}
