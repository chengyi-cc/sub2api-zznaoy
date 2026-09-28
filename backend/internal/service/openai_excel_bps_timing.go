package service

import (
	"context"
	"crypto/tls"
	"errors"
	"net/http"
	"net/http/httptrace"
	"sync"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/ctxkey"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/service/basispoints"
	"github.com/tidwall/gjson"
	"go.uber.org/zap"
)

// Observations contain only locally chosen names, times and counters. They never
// retain prompts, tool arguments, credentials, response bodies or error strings.
// The mutex protects cancellation racing with the stream conversion goroutine.
type excelBPSRequestTiming struct {
	mu     sync.Mutex
	origin time.Time
	first  map[string]time.Time
	last   map[string]time.Time
	values map[string]any
}

func newExcelBPSRequestTiming(ctx context.Context, start time.Time, enabled bool) *excelBPSRequestTiming {
	if !enabled {
		return nil
	}
	if ingress, ok := ctx.Value(ctxkey.RequestStartTime).(time.Time); ok && !ingress.IsZero() && !ingress.After(start) {
		start = ingress
	}
	t := &excelBPSRequestTiming{origin: start, first: make(map[string]time.Time), last: make(map[string]time.Time), values: make(map[string]any)}
	t.mark("forward_started")
	return t
}

func (t *excelBPSRequestTiming) mark(stage string) {
	if t != nil {
		t.markAt(stage, time.Now())
	}
}

func (t *excelBPSRequestTiming) markAt(stage string, now time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if _, ok := t.first[stage]; !ok {
		t.first[stage] = now
	}
	t.last[stage] = now
	t.values[stage+"_ms"] = t.first[stage].Sub(t.origin).Milliseconds()
	for _, counter := range [][2]string{{"repair_started", "tool_repairs"}, {"compaction_completed", "compactions_completed"}} {
		if stage == counter[0] {
			n, _ := t.values[counter[1]].(int)
			t.values[counter[1]] = n + 1
		}
	}
	for _, pair := range [][3]string{{"compaction_started", "compaction_completed", "compaction_duration_ms"}, {"repair_started", "repair_completed", "repair_duration_ms"}, {"validation_started", "validation_completed", "validation_duration_ms"}} {
		if stage == pair[1] {
			if since, ok := t.last[pair[0]]; ok && !now.Before(since) {
				key := pair[2]
				ms, _ := t.values[key].(int64)
				t.values[key] = ms + now.Sub(since).Milliseconds()
				delete(t.last, pair[0])
			}
		}
	}
}

func (t *excelBPSRequestTiming) observe(stage basispoints.StreamStage) { t.mark(string(stage)) }

// Called only around model HTTP requests, not attachment transfers. These are
// time-to-headers measurements, not time-to-first-token or model compute time.
func (t *excelBPSRequestTiming) beginHTTP(req *http.Request) (*http.Request, func(*http.Response, error)) {
	if t == nil {
		return req, func(*http.Response, error) {}
	}
	started := time.Now()
	t.markAt("http_started", started)
	var transportMu sync.Mutex
	transport := make(map[string]any)
	mark := func(key string) {
		transportMu.Lock()
		defer transportMu.Unlock()
		if _, exists := transport[key]; !exists {
			transport[key] = time.Since(started).Milliseconds()
		}
	}
	trace := &httptrace.ClientTrace{
		GetConn:      func(string) { mark("get_connection_ms") },
		DNSStart:     func(httptrace.DNSStartInfo) { mark("dns_started_ms") },
		DNSDone:      func(httptrace.DNSDoneInfo) { mark("dns_completed_ms") },
		ConnectStart: func(string, string) { mark("connect_started_ms") },
		ConnectDone: func(_, _ string, err error) {
			if err == nil {
				mark("connect_completed_ms")
			}
		},
		TLSHandshakeStart: func() { mark("tls_started_ms") },
		TLSHandshakeDone: func(_ tls.ConnectionState, err error) {
			if err == nil {
				mark("tls_completed_ms")
			}
		},
		GotConn: func(info httptrace.GotConnInfo) {
			mark("connection_obtained_ms")
			transportMu.Lock()
			transport["connection_reused"] = info.Reused
			transportMu.Unlock()
		},
		WroteRequest: func(info httptrace.WroteRequestInfo) {
			if info.Err == nil {
				mark("request_written_ms")
			}
		},
		GotFirstResponseByte: func() { mark("first_response_byte_ms") },
	}
	req = req.WithContext(httptrace.WithClientTrace(req.Context(), trace))
	return req, func(resp *http.Response, err error) {
		now := time.Now()
		transportMu.Lock()
		item := make(map[string]any, len(transport)+2)
		for key, value := range transport {
			item[key] = value
		}
		transportMu.Unlock()
		item["wait_ms"] = now.Sub(started).Milliseconds()
		item["transport_error"] = err != nil
		t.mu.Lock()
		defer t.mu.Unlock()
		n, _ := t.values["http_attempts"].(int)
		t.values["http_attempts"] = n + 1
		attempts, _ := t.values["http_transport"].([]map[string]any)
		t.values["http_transport"] = append(attempts, item)
		ms, _ := t.values["http_wait_total_ms"].(int64)
		t.values["http_wait_total_ms"] = ms + now.Sub(started).Milliseconds()
		if resp != nil && err == nil {
			if _, ok := t.values["first_headers_ms"]; !ok {
				t.values["first_headers_ms"] = now.Sub(t.origin).Milliseconds()
			}
			t.values["last_http_status"] = resp.StatusCode
		}
	}
}

func (t *excelBPSRequestTiming) compactionPolicy(body []byte, callerExplicit bool) {
	if t == nil {
		return
	}
	thresholds := []int64{}
	for _, entry := range gjson.GetBytes(body, "context_management").Array() {
		if entry.Get("type").String() == "compaction" {
			thresholds = append(thresholds, entry.Get("compact_threshold").Int())
		}
	}
	trigger := false
	for _, entry := range gjson.GetBytes(body, "input").Array() {
		if entry.Get("type").String() == "compaction_trigger" {
			trigger = true
		}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.values["compaction_thresholds"] = thresholds
	t.values["compaction_policy_from_caller"] = callerExplicit
	t.values["compaction_trigger"] = trigger
}

func (t *excelBPSRequestTiming) snapshot(now time.Time) map[string]any {
	t.mu.Lock()
	defer t.mu.Unlock()
	fields := make(map[string]any, len(t.values)+8)
	for key, value := range t.values {
		fields[key] = value
	}
	fields["elapsed_ms"] = now.Sub(t.origin).Milliseconds()
	for _, interval := range [][3]string{
		{"prepare_duration_ms", "forward_started", "prepared"},
		{"auth_duration_ms", "auth_started", "auth_completed"},
		{"attachment_duration_ms", "attachments_started", "attachments_completed"},
		{"first_tool_buffer_ms", "tool_ready", "tool_emitted"},
	} {
		from, okFrom := t.first[interval[1]]
		to, okTo := t.first[interval[2]]
		if okFrom && okTo && !to.Before(from) {
			fields[interval[0]] = to.Sub(from).Milliseconds()
		}
	}
	if from, ok := t.last["message_completed"]; ok {
		if to, ok := t.first["upstream_completed"]; ok && !to.Before(from) {
			fields["last_message_to_completion_ms"] = to.Sub(from).Milliseconds()
		}
	}
	return fields
}

func (t *excelBPSRequestTiming) finish(ctx context.Context, accountID int64, result *OpenAIForwardResult, err error) {
	if t == nil {
		return
	}
	fields := t.snapshot(time.Now())
	fields["succeeded"] = err == nil && result != nil
	fields["canceled"] = errors.Is(err, context.Canceled) || errors.Is(ctx.Err(), context.Canceled)
	logger.FromContext(ctx).Info("excel_bps_request_timing", zap.String("component", "service.openai_excel_bps"), zap.Int64("account_id", accountID), zap.Any("timing", fields))
}
