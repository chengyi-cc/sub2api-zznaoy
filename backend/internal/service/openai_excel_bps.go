package service

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"sync/atomic"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/errorarchive"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/pkg/openai"
	"github.com/Wei-Shaw/sub2api/internal/service/basispoints"
	"github.com/Wei-Shaw/sub2api/internal/util/logredact"
	"github.com/Wei-Shaw/sub2api/internal/util/transportdiag"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

var excelBPSReplay basispoints.ReplayCache
var excelBPSCatalog basispoints.CatalogCache

func (s *OpenAIGatewayService) disableExcelBPSOn403(ctx context.Context, account *Account) bool {
	if !account.IsExcelBPSAutoDisableOn403Enabled() {
		return false
	}
	repo, ok := s.accountRepo.(AccountExcelBPSRepository)
	if !ok {
		return false
	}
	stateCtx, cancel := openAIAccountStateContext(ctx)
	defer cancel()
	changed, err := repo.DisableExcelBPSOn403(stateCtx, account)
	if err != nil {
		// Do not log upstream bodies, credentials or database query arguments.
		logger.LegacyPrintf("service.openai_excel_bps", "auto-disable failed: account_id=%d error_type=%T", account.ID, err)
		return false
	}
	if changed {
		logger.LegacyPrintf("service.openai_excel_bps", "automatically disabled Excel BPS after upstream HTTP 403: account_id=%d", account.ID)
	}
	return changed
}

func excelBPSAccountID(account *Account, accessToken string) string {
	if accountID := strings.TrimSpace(account.GetChatGPTAccountID()); accountID != "" {
		return accountID
	}
	claims, err := openai.DecodeIDToken(accessToken)
	if err != nil || claims.OpenAIAuth == nil {
		return ""
	}
	return strings.TrimSpace(claims.OpenAIAuth.ChatGPTAccountID)
}

func newExcelBPSRequest(ctx context.Context, body []byte, token, accountID string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, basispoints.ResponsesURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header = http.Header{
		"Authorization": {"Bearer " + token}, "Chatgpt-Account-Id": {accountID}, "X-Openai-Account-Id": {accountID},
		"X-Basispoints-Auth-Mode": {"chatgpt"}, "Content-Type": {"application/json"}, "Accept": {"text/event-stream"},
		"Origin": {"https://bps.openai.com"}, "User-Agent": {"Mozilla/5.0"},
		"X-Openai-Internal-Basispoints-Client-Product":       {"basispoints-excel-plugin"},
		"X-Openai-Internal-Basispoints-Client-Agent-Profile": {"excel"},
	}
	// The bridge owns explicit corrective continuations; transport must not
	// rewind a possibly delivered model request after a connection failure.
	req.GetBody = nil
	return (&transportdiag.Trace{}).Request(req), nil
}

// BPS deliberately bypasses Codex ticket/cookie injection and OAuth plugins:
// only the selected account's bearer and ChatGPT account ID belong on this host.
func (s *OpenAIGatewayService) forwardExcelBPS(ctx context.Context, c *gin.Context, account *Account, body []byte, start time.Time) (out *OpenAIForwardResult, outErr error) {
	timing := newExcelBPSRequestTiming(ctx, start, s.cfg != nil && s.cfg.Gateway.ExcelBPS.LogRequestTiming)
	defer func() { timing.finish(ctx, account.ID, out, outErr) }()
	// Generated only on failure. Healthy requests do not pay for a second JSON
	// traversal, and long image histories do not crowd out correction evidence.
	var diagnosticSummary json.RawMessage
	archiveDiagnostic := func(phase string, response []byte) {
		if !errorarchive.HasTrace(ctx) {
			return
		}
		if diagnosticSummary == nil {
			diagnosticSummary = errorarchive.RequestSummary(body)
		}
		errorarchive.AddDiagnosticSummary(ctx, phase, diagnosticSummary, response)
	}
	fail := func(status int, code, message string, param ...string) (*OpenAIForwardResult, error) {
		failureParam := ""
		if len(param) > 0 {
			failureParam = param[0]
		}
		if diagnosticSummary == nil {
			raw, _ := json.Marshal(map[string]any{"code": code, "message": message})
			archiveDiagnostic("request_failure", raw)
		}
		// A compact keepalive may already have committed SSE headers. Otherwise
		// finish a single JSON response so the handler cannot append another error.
		committed := StopOpenAICompactSSEKeepaliveCommitted(c)
		MarkResponseCommitted(c)
		if committed {
			writeOpenAICompactSSEFailureMessageParam(c, status, code, message, failureParam)
		} else {
			errorType := "invalid_request_error"
			if status >= 500 {
				errorType = "server_error"
			}
			errorBody := gin.H{"type": errorType, "code": code, "message": message}
			if failureParam != "" {
				errorBody["param"] = failureParam
			}
			c.JSON(status, gin.H{"error": errorBody})
		}
		return nil, fmt.Errorf("excel BPS: %s", code)
	}
	originalModel := gjson.GetBytes(body, "model").String()
	model := account.GetMappedModel(originalModel)
	// The handler strips stream from normalized compact bodies but retains the
	// client's original intent in context. Honor it when returning BPS events.
	stream := gjson.GetBytes(body, "stream").Bool() || openAICompactClientWantsStream(c)
	clientCanceled := func() (*OpenAIForwardResult, error) {
		StopOpenAICompactSSEKeepaliveCommitted(c)
		MarkResponseCommitted(c)
		MarkOpsClientCancellation(c, stream)
		return nil, context.Canceled
	}
	var err error
	body, err = sjson.SetBytes(body, "model", model)
	if err != nil {
		return fail(400, "basispoints_request_invalid", "Invalid model request")
	}
	identity, threadID := resolveOpenAIWSExecutionScope(c, body, getAPIKeyIDFromContext(c))
	if identity != "" {
		body, err = sjson.SetBytes(body, "prompt_cache_key", identity)
		if err != nil {
			return nil, err
		}
	}
	if isOpenAIResponsesCompactPath(c) {
		var request map[string]any
		decoder := json.NewDecoder(bytes.NewReader(body))
		decoder.UseNumber() // Preserve large integer tool arguments in compact history.
		if err = decoder.Decode(&request); err != nil {
			return fail(400, "basispoints_request_invalid", "Invalid compact request")
		}
		var input []any
		switch v := request["input"].(type) {
		case []any:
			input = v
		case string:
			input = []any{map[string]any{"role": "user", "content": v}}
		default:
			return fail(400, "basispoints_request_invalid", "Compact requires input")
		}
		request["input"] = append(input, map[string]any{"type": "compaction_trigger"})
		request["tool_choice"] = "none"
		body, err = json.Marshal(request)
		if err != nil {
			return nil, err
		}
	}
	scope := fmt.Sprintf("account:%d/key:%d/thread:%s", account.ID, getAPIKeyIDFromContext(c), identity)
	replay := &excelBPSReplay
	if identity == "" {
		// Without a declared conversation identity only complete caller-supplied
		// history may be restored. Never share result-only replay across requests.
		replay = nil
	}
	imageBody, imagePlan, err := prepareExcelBPSImages(body)
	if err != nil {
		return fail(400, "basispoints_image_invalid", err.Error())
	}
	// Thread IDs distinguish sibling agents which may share a session header.
	// Without an explicit caller key, thread and upstream owner, do not inherit.
	var catalog *basispoints.CatalogCache
	catalogScope := scope
	if owner := strings.TrimSpace(account.GetChatGPTAccountID()); owner != "" && threadID != "" && getAPIKeyIDFromContext(c) > 0 && account.ID > 0 {
		catalog = &excelBPSCatalog
		catalogScope += "/owner:" + openAIEncryptedContentDigest(owner)
	}
	options := basispoints.PrepareOptions{}
	if s.cfg != nil {
		options.CompactionThresholdTokens = s.cfg.Gateway.ExcelBPS.CompactionThresholdTokens
	}
	upstreamBody, bridge, err := basispoints.PrepareWithCatalogOptions(imageBody, scope, replay, catalog, catalogScope, options)
	if err != nil {
		var contentErr *basispoints.ContentValidationError
		if errors.As(err, &contentErr) {
			return fail(400, "basispoints_request_invalid", err.Error(), contentErr.Path)
		}
		return fail(400, "basispoints_request_invalid", err.Error())
	}
	timing.mark("prepared")
	if timing != nil {
		timing.compactionPolicy(upstreamBody, gjson.GetBytes(imageBody, "context_management").IsArray())
		bridge.ObserveLifecycle(timing.observe)
	}
	timing.mark("auth_started")
	token, _, err := s.GetAccessToken(ctx, account)
	timing.mark("auth_completed")
	if err != nil {
		if isExcelBPSClientCancellation(c, err) {
			return clientCanceled()
		}
		return fail(502, "basispoints_auth_unavailable", "Account OAuth credential is unavailable")
	}
	accountID := excelBPSAccountID(account, token)
	if accountID == "" {
		return fail(400, "basispoints_account_id_missing", "Excel BPS requires chatgpt_account_id")
	}
	// Keep recovery state separate from ordinary Responses and isolate it by
	// account identity, API key, thread and target model. No identity, no reuse.
	encryptedScope := ""
	if identity != "" {
		encryptedScope = "excel-bps:" + scope + "/owner:" + openAIEncryptedContentDigest(accountID+"\x00"+model)
		invalid := s.sessionInvalidEncryptedContentDigests(0, encryptedScope)
		upstreamBody, _ = excelBPSDropRejectedReasoning(upstreamBody, invalid)
	}
	imageScope := excelBPSAttachmentScope{account.ID, getAPIKeyIDFromContext(c), accountID}
	timing.mark("attachments_started")
	upstreamBody, err = s.excelBPSImages.upload(ctx, upstreamBody, imagePlan, imageScope, token, account, s.httpUpstream)
	timing.mark("attachments_completed")
	if err != nil {
		if isExcelBPSClientCancellation(c, err) {
			return clientCanceled()
		}
		var uploadErr *excelBPSAttachmentHTTPError
		if errors.As(err, &uploadErr) && uploadErr.status == http.StatusUnauthorized {
			s.handleExcelBPSUnauthorized(ctx, account, uploadErr.status, uploadErr.header, uploadErr.body)
			return fail(http.StatusUnauthorized, "basispoints_image_upload_failed", "Excel BPS attachment authentication failed; request was not replayed")
		}
		return fail(502, "basispoints_image_upload_failed", err.Error())
	}
	requestCtx := WithHTTPUpstreamRedirectsDisabled(WithHTTPUpstreamProfile(ctx, HTTPUpstreamProfileExcelBPS))
	req, err := newExcelBPSRequest(requestCtx, upstreamBody, token, accountID)
	if err != nil {
		return nil, err
	}
	proxyURL := ""
	if account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}
	SetActualOpenAIUpstreamEndpoint(c, "/basispoints/api/responses")
	SetOpsUpstreamModel(c, model)
	sent := time.Now()
	var activeTransportRequest atomic.Pointer[http.Request]
	req, finishHTTP := timing.beginHTTP(req)
	activeTransportRequest.Store(req)
	resp, err := s.httpUpstream.Do(req, proxyURL, account.ID, account.Concurrency)
	finishHTTP(resp, err)
	SetOpsLatencyMs(c, OpsUpstreamLatencyMsKey, time.Since(sent).Milliseconds())
	if err != nil {
		if isExcelBPSClientCancellation(c, err) {
			return clientCanceled()
		}
		archiveDiagnostic("upstream_transport", recordExcelBPSTransportFailure(c, account, req, err))
		return fail(502, "basispoints_transport_error", "Excel BPS connection failed; request was not replayed")
	}
	var recoveredDigests []string
	if resp.StatusCode == http.StatusBadRequest {
		rejection, readErr := io.ReadAll(io.LimitReader(resp.Body, 512<<10))
		if isExcelBPSClientCancellation(c, readErr) {
			_ = resp.Body.Close()
			return clientCanceled()
		}
		archiveDiagnostic("upstream_rejection", rejection)
		_ = resp.Body.Close()
		resp.Body = io.NopCloser(bytes.NewReader(rejection))
		if readErr == nil && ctx.Err() == nil {
			retryBody, digests := excelBPSRejectedReasoningRetry(upstreamBody, rejection)
			if len(digests) != 0 {
				retryReq, buildErr := newExcelBPSRequest(requestCtx, retryBody, token, accountID)
				if buildErr == nil {
					_ = resp.Body.Close()
					logger.LegacyPrintf("service.openai_excel_bps", "retrying once after rejected optional reasoning: account_id=%d digests=%d", account.ID, len(digests))
					upstreamBody, recoveredDigests = retryBody, digests
					retryReq, finishHTTP = timing.beginHTTP(retryReq)
					activeTransportRequest.Store(retryReq)
					resp, err = s.httpUpstream.Do(retryReq, proxyURL, account.ID, account.Concurrency)
					finishHTTP(resp, err)
					SetOpsLatencyMs(c, OpsUpstreamLatencyMsKey, time.Since(sent).Milliseconds())
					if err != nil {
						if isExcelBPSClientCancellation(c, err) {
							return clientCanceled()
						}
						archiveDiagnostic("recovery_transport", recordExcelBPSTransportFailure(c, account, retryReq, err))
						return fail(502, "basispoints_transport_error", "Excel BPS recovery connection failed; no further retry was attempted")
					}
				}
			}
		}
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		raw, readErr := io.ReadAll(io.LimitReader(resp.Body, 512<<10))
		if isExcelBPSClientCancellation(c, readErr) {
			return clientCanceled()
		}
		archiveDiagnostic("upstream_http_error", raw)
		s.handleExcelBPSUnauthorized(ctx, account, resp.StatusCode, resp.Header, raw)
		s.excelBPSImages.invalidateRejected(imageScope, upstreamBody, raw)
		if resp.StatusCode == http.StatusTooManyRequests && s.rateLimitService != nil {
			stateCtx, cancel := openAIAccountStateContext(ctx)
			s.rateLimitService.handle429Cooldown(stateCtx, account, resp.Header, raw)
			cancel()
		}
		// Preserve the original rejection for Ops without exposing it to clients.
		// BPS errors can echo request fields, so redact before storing diagnostics.
		upstreamMessage := fmt.Sprintf("Excel BPS returned HTTP %d", resp.StatusCode)
		upstreamDetail := ""
		if s.cfg != nil && s.cfg.Gateway.LogUpstreamErrorBody {
			safeBody := excelBPSSanitizeErrorBody(string(raw), token, account)
			maxBytes := s.cfg.Gateway.LogUpstreamErrorBodyMaxBytes
			if maxBytes <= 0 {
				maxBytes = 2048
			}
			upstreamDetail, _ = sanitizeErrorBodyForStorage(safeBody, maxBytes)
			if message := strings.TrimSpace(extractUpstreamErrorMessage([]byte(safeBody))); message != "" {
				upstreamMessage = truncateString(message, 2048)
			}
		}
		if resp.StatusCode == http.StatusBadRequest && gjson.GetBytes(raw, "error.code").String() == "invalid_encrypted_content" {
			if upstreamDetail == "" || !json.Valid([]byte(upstreamDetail)) {
				upstreamDetail = `{"error":{"code":"invalid_encrypted_content"}}`
			}
			if detail, detailErr := sjson.SetRaw(upstreamDetail, "gateway_diagnostics.encrypted_categories", excelBPSEncryptedHistoryDiagnostic(upstreamBody)); detailErr == nil {
				upstreamDetail = detail
			}
			if detail, detailErr := sjson.Set(upstreamDetail, "gateway_diagnostics.recovery_attempted", len(recoveredDigests) != 0); detailErr == nil {
				upstreamDetail = detail
			}
		}
		setOpsUpstreamError(c, resp.StatusCode, upstreamMessage, upstreamDetail)
		appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
			Platform: account.Platform, AccountID: account.ID, AccountName: account.Name,
			ProxyID: opsUpstreamProxyID(account), ProxyName: opsUpstreamProxyName(account),
			UpstreamStatusCode: resp.StatusCode, UpstreamRequestID: resp.Header.Get("x-request-id"),
			UpstreamURL: basispoints.ResponsesURL, Kind: "http_error",
			Message: upstreamMessage, Detail: upstreamDetail, UpstreamResponseBody: upstreamDetail,
		})
		code := gjson.GetBytes(raw, "error.code").String()
		if resp.StatusCode == http.StatusBadRequest && code == "invalid_encrypted_content" {
			logger.LegacyPrintf("service.openai_excel_bps", "encrypted history rejected: account_id=%d retry_attempted=%t categories=%s", account.ID, len(recoveredDigests) != 0, excelBPSEncryptedHistoryDiagnostic(upstreamBody))
			return fail(resp.StatusCode, code, "Encrypted conversation history could not be verified. Automatic recovery was unavailable or unsuccessful; restore the original history or start a new conversation with a saved summary.")
		}
		if code == "basispoints_model_access_changed" {
			return fail(resp.StatusCode, code, "This model is not available on the account's Excel BPS endpoint")
		}
		if resp.StatusCode == http.StatusUnauthorized {
			return fail(resp.StatusCode, "basispoints_upstream_error", "Excel BPS authentication failed; request was not replayed")
		}
		message := "Excel BPS rejected this request; account scheduling was not changed"
		if resp.StatusCode == http.StatusTooManyRequests {
			message = "Excel BPS rate limit exceeded; request was not replayed"
		}
		if resp.StatusCode == http.StatusForbidden && s.disableExcelBPSOn403(ctx, account) {
			message = "Excel BPS rejected this request; Excel BPS was automatically disabled for this account; request was not replayed"
		}
		return fail(resp.StatusCode, "basispoints_upstream_error", message)
	}
	s.UpdateCodexUsageSnapshotFromHeaders(ctx, account.ID, resp.Header)
	repairBody := upstreamBody
	if errorarchive.HasTrace(ctx) {
		failureIndex := 0
		bridge.ObserveToolFailure(func(failed map[string]any, validation error) {
			phase := "tool_validation"
			if failureIndex > 0 {
				phase = "tool_correction_rejected"
			}
			failureIndex++
			// Upstream responses may echo the entire instructions/catalog. Keep
			// the actual tool calls and usage so both failed attempts fit.
			compact := map[string]any{"id": failed["id"], "status": failed["status"], "model": failed["model"], "usage": failed["usage"]}
			var calls []any
			if output, ok := failed["output"].([]any); ok {
				for _, value := range output {
					if item, ok := value.(map[string]any); ok && (item["type"] == "function_call" || item["type"] == "custom_tool_call") {
						calls = append(calls, item)
					}
				}
			}
			compact["output"] = calls
			raw, _ := json.Marshal(map[string]any{"response": compact, "validation_error": validation.Error(), "attempt": failureIndex - 1})
			archiveDiagnostic(phase, raw)
		})
	}
	maxLineSize := defaultMaxLineSize
	if s.cfg != nil && s.cfg.Gateway.MaxLineSize > 0 {
		maxLineSize = s.cfg.Gateway.MaxLineSize
	}
	bridge.SetSSEMaxBytes(maxLineSize)
	converted := bridge.StreamWithToolRepair(requestCtx, resp.Body, func(repairCtx context.Context, failed map[string]any, validation error) (map[string]any, error) {
		correctedBody, buildErr := basispoints.BuildToolRepairRequest(repairBody, failed, validation)
		if buildErr != nil {
			return nil, buildErr
		}
		// Limit the total time spent on an individual corrective continuation.
		repairCtx, cancel := context.WithTimeout(repairCtx, 45*time.Second)
		defer cancel()
		repairReq, buildErr := newExcelBPSRequest(repairCtx, correctedBody, token, accountID)
		if buildErr != nil {
			return nil, buildErr
		}
		repairReq, finishRepairHTTP := timing.beginHTTP(repairReq)
		activeTransportRequest.Store(repairReq)
		repairResp, callErr := s.httpUpstream.Do(repairReq, proxyURL, account.ID, account.Concurrency)
		finishRepairHTTP(repairResp, callErr)
		if callErr != nil {
			if repairCtx.Err() != nil {
				return nil, repairCtx.Err()
			}
			archiveDiagnostic("correction_transport", excelBPSTransportDiagnostic(repairReq, callErr))
			return nil, fmt.Errorf("excel BPS correction connection failed (%s)", transportdiag.Classify(callErr))
		}
		defer func() { _ = repairResp.Body.Close() }()
		stop := context.AfterFunc(repairCtx, func() { _ = repairResp.Body.Close() })
		defer stop()
		if repairResp.StatusCode < 200 || repairResp.StatusCode >= 300 {
			raw, _ := io.ReadAll(io.LimitReader(repairResp.Body, 512<<10))
			archiveDiagnostic("tool_correction_http_error", raw)
			s.handleExcelBPSUnauthorized(repairCtx, account, repairResp.StatusCode, repairResp.Header, raw)
			if repairResp.StatusCode == http.StatusTooManyRequests && s.rateLimitService != nil {
				stateCtx, stateCancel := openAIAccountStateContext(repairCtx)
				s.rateLimitService.handle429Cooldown(stateCtx, account, repairResp.Header, raw)
				stateCancel()
			}
			if repairResp.StatusCode == http.StatusForbidden {
				s.disableExcelBPSOn403(repairCtx, account)
			}
			return nil, fmt.Errorf("excel BPS correction returned HTTP %d", repairResp.StatusCode)
		}
		s.UpdateCodexUsageSnapshotFromHeaders(repairCtx, account.ID, repairResp.Header)
		repairBody = correctedBody
		return basispoints.ReadToolRepairResponse(repairResp.Body)
	})
	defer func() { _ = converted.Close() }()
	requestedEffort := coalesceRequestedReasoningEffort(RequestedReasoningEffortFromContext(ctx), &bridge.RequestedEffort)
	result := &OpenAIForwardResult{Model: originalModel, UpstreamModel: model, UpstreamEndpoint: "/basispoints/api/responses", Stream: stream, ReasoningEffort: &bridge.Effort, RequestedReasoningEffort: requestedEffort, RequestID: resp.Header.Get("x-request-id")}
	if stream {
		// Take ownership before writing events so the compact heartbeat cannot
		// interleave writes or leave a committed SSE response followed by JSON.
		StopOpenAICompactSSEKeepaliveCommitted(c)
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("X-Accel-Buffering", "no")
	}
	scanner := newOpenAISSEReadPump(converted, maxLineSize)
	defer scanner.Close()
	heartbeat := time.NewTicker(15 * time.Second)
	defer heartbeat.Stop()
	keepalive := func() {
		if stream && ctx.Err() == nil {
			_, _ = c.Writer.WriteString(": keepalive\n\n")
			c.Writer.Flush()
		}
	}
	var completed []byte
	terminal := ""
	for scanner.Next(ctx, 0, heartbeat.C, keepalive) {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			payload := []byte(strings.TrimPrefix(line, "data: "))
			kind := gjson.GetBytes(payload, "type").String()
			s.parseSSEUsageBytes(payload, &result.Usage)
			if account.IsExcelBPSCacheCreationAsInputEnabled() {
				payload, err = excelBPSDownstreamUsage(payload)
				if err != nil {
					break
				}
				line = "data: " + string(payload)
			}
			if result.FirstTokenMs == nil && (kind == "response.output_text.delta" || kind == "response.output_item.added") {
				ms := int(time.Since(start).Milliseconds())
				result.FirstTokenMs = &ms
			}
			switch kind {
			case "response.completed", "response.failed", "response.incomplete", "error":
				terminal = kind
				completed = []byte(gjson.GetBytes(payload, "response").Raw)
				result.ResponseID = gjson.GetBytes(payload, "response.id").String()
				result.UpstreamResponseModel = gjson.GetBytes(payload, "response.model").String()
			}
		}
		if stream {
			if _, err = c.Writer.WriteString(line + "\n"); err != nil {
				result.streamReadIncomplete = true
				result.ClientDisconnect = true
				if isExcelBPSClientCancellation(c, c.Request.Context().Err()) {
					MarkOpsClientCancellation(c, stream)
				}
				result.Duration = time.Since(start)
				return result, err
			}
			if line == "" {
				c.Writer.Flush()
			}
		}
	}
	result.Duration = time.Since(start)
	result.UpstreamTerminalEvent = terminal
	if err = scanner.Err(); err != nil || terminal == "" {
		result.streamReadIncomplete = true
		if isExcelBPSClientCancellation(c, ctx.Err()) {
			StopOpenAICompactSSEKeepaliveCommitted(c)
			MarkResponseCommitted(c)
			MarkOpsClientCancellation(c, stream)
			result.ClientDisconnect = true
			return result, ctx.Err()
		}
		if ctx.Err() != nil {
			err = ctx.Err()
		}
		archiveDiagnostic("stream_transport", recordExcelBPSTransportFailure(c, account, activeTransportRequest.Load(), err))
		MarkResponseCommitted(c)
		if stream {
			writeOpenAICompactSSEFailureMessage(c, 502, "basispoints_stream_incomplete", "Upstream stream ended before completion")
		} else {
			c.JSON(502, gin.H{"error": gin.H{"code": "basispoints_stream_incomplete", "message": "Excel BPS stream ended before completion"}})
		}
		return result, fmt.Errorf("excel BPS stream incomplete")
	}
	if terminal != "response.completed" {
		MarkResponseCommitted(c)
	}
	if !stream {
		if terminal != "response.completed" {
			c.JSON(502, gin.H{"error": gin.H{"code": "basispoints_protocol_error", "message": "Excel BPS did not complete the response"}})
		} else {
			c.Data(200, "application/json", completed)
		}
	}
	if terminal != "response.completed" {
		return result, fmt.Errorf("excel BPS terminal: %s", terminal)
	}
	// A rejected request alone does not prove recovery works. Remember only
	// digests whose removal led to a fully completed response.
	if encryptedScope != "" && len(recoveredDigests) != 0 {
		s.markOpenAIWSInvalidEncryptedContentLineage(0, encryptedScope, recoveredDigests)
	}
	s.bindHTTPResponseAccount(ctx, c, account, result.ResponseID)
	return result, nil
}

var excelBPSBearerPattern = regexp.MustCompile(`(?i)\bBearer\s+[^\s"',;<>]+`)
var excelBPSURLCredentialsPattern = regexp.MustCompile(`(https?://)[^/\s@]+@`)

func excelBPSSanitizeErrorBody(raw, token string, account *Account) string {
	if !json.Valid([]byte(raw)) {
		return ""
	}
	secrets := append([]string{token}, excelBPSAccountSecrets(account)...)
	fields := make(map[string]string)
	for _, key := range []string{"message", "code", "type", "param"} {
		value := gjson.Get(raw, "error."+key)
		if value.Type != gjson.String {
			continue
		}
		clean := value.String()
		for _, secret := range secrets {
			if secret != "" {
				clean = strings.ReplaceAll(clean, secret, "[redacted]")
			}
		}
		clean = excelBPSBearerPattern.ReplaceAllString(clean, "Bearer [redacted]")
		clean = excelBPSURLCredentialsPattern.ReplaceAllString(clean, "${1}[redacted]@")
		clean = sanitizeUpstreamErrorMessage(clean)
		fields[key] = truncateString(logredact.RedactText(clean, "authorization", "api_key", "apikey", "token", "secret", "key", "cookie", "ticket", "recovery_ticket", "x-amz-signature", "x-amz-credential", "x-amz-security-token"), 2048)
	}
	encoded, _ := json.Marshal(map[string]any{"error": fields})
	return string(encoded)
}

func excelBPSAccountSecrets(account *Account) []string {
	var secrets []string
	for _, key := range []string{"access_token", "refresh_token", "id_token", "api_key", "session_key", "cookie"} {
		if value := account.GetCredential(key); value != "" {
			secrets = append(secrets, value)
		}
	}
	if account.Proxy != nil && account.Proxy.Password != "" {
		secrets = append(secrets, account.Proxy.Password)
	}
	return secrets
}
