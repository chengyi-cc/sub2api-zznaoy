package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/errorarchive"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/pkg/openai"
	"github.com/Wei-Shaw/sub2api/internal/service/basispoints"
	"github.com/Wei-Shaw/sub2api/internal/util/logredact"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

var excelBPSReplay basispoints.ReplayCache

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
	return req, nil
}

// BPS deliberately bypasses Codex ticket/cookie injection and OAuth plugins:
// only the selected account's bearer and ChatGPT account ID belong on this host.
func (s *OpenAIGatewayService) forwardExcelBPS(ctx context.Context, c *gin.Context, account *Account, body []byte, start time.Time) (*OpenAIForwardResult, error) {
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
	fail := func(status int, code, message string) (*OpenAIForwardResult, error) {
		if diagnosticSummary == nil {
			raw, _ := json.Marshal(map[string]any{"code": code, "message": message})
			archiveDiagnostic("request_failure", raw)
		}
		// A compact keepalive may already have committed SSE headers. Otherwise
		// finish a single JSON response so the handler cannot append another error.
		committed := StopOpenAICompactSSEKeepaliveCommitted(c)
		MarkResponseCommitted(c)
		if committed {
			writeOpenAICompactSSEFailureMessage(c, status, code, message)
		} else {
			c.JSON(status, gin.H{"error": gin.H{"type": "invalid_request_error", "code": code, "message": message}})
		}
		return nil, fmt.Errorf("excel BPS: %s", code)
	}
	originalModel := gjson.GetBytes(body, "model").String()
	model := account.GetMappedModel(originalModel)
	// The handler strips stream from normalized compact bodies but retains the
	// client's original intent in context. Honor it when returning BPS events.
	stream := gjson.GetBytes(body, "stream").Bool() || openAICompactClientWantsStream(c)
	var err error
	body, err = sjson.SetBytes(body, "model", model)
	if err != nil {
		return fail(400, "basispoints_request_invalid", "Invalid model request")
	}
	identity, _ := resolveOpenAIWSExecutionScope(c, body, getAPIKeyIDFromContext(c))
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
	upstreamBody, bridge, err := basispoints.Prepare(imageBody, scope, replay)
	if err != nil {
		return fail(400, "basispoints_request_invalid", err.Error())
	}
	token, _, err := s.GetAccessToken(ctx, account)
	if err != nil {
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
	upstreamBody, err = s.excelBPSImages.upload(ctx, upstreamBody, imagePlan, imageScope, token, account, s.httpUpstream)
	if err != nil {
		return fail(502, "basispoints_image_upload_failed", err.Error())
	}
	requestCtx := WithHTTPUpstreamRedirectsDisabled(WithHTTPUpstreamProfile(ctx, HTTPUpstreamProfileLongStream))
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
	resp, err := s.httpUpstream.Do(req, proxyURL, account.ID, account.Concurrency)
	SetOpsLatencyMs(c, OpsUpstreamLatencyMsKey, time.Since(sent).Milliseconds())
	if err != nil {
		return fail(502, "basispoints_transport_error", "Excel BPS connection failed; request was not replayed")
	}
	var recoveredDigests []string
	if resp.StatusCode == http.StatusBadRequest {
		rejection, readErr := io.ReadAll(io.LimitReader(resp.Body, 512<<10))
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
					resp, err = s.httpUpstream.Do(retryReq, proxyURL, account.ID, account.Concurrency)
					SetOpsLatencyMs(c, OpsUpstreamLatencyMsKey, time.Since(sent).Milliseconds())
					if err != nil {
						return fail(502, "basispoints_transport_error", "Excel BPS recovery connection failed; no further retry was attempted")
					}
				}
			}
		}
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 512<<10))
		archiveDiagnostic("upstream_http_error", raw)
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
		repairResp, callErr := s.httpUpstream.Do(repairReq, proxyURL, account.ID, account.Concurrency)
		if callErr != nil {
			if repairCtx.Err() != nil {
				return nil, repairCtx.Err()
			}
			return nil, fmt.Errorf("excel BPS correction connection failed")
		}
		defer func() { _ = repairResp.Body.Close() }()
		stop := context.AfterFunc(repairCtx, func() { _ = repairResp.Body.Close() })
		defer stop()
		if repairResp.StatusCode < 200 || repairResp.StatusCode >= 300 {
			raw, _ := io.ReadAll(io.LimitReader(repairResp.Body, 512<<10))
			archiveDiagnostic("tool_correction_http_error", raw)
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
	scanner := newOpenAISSEReadPump(converted, 16<<20)
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
		if ctx.Err() != nil {
			result.ClientDisconnect = true
			return result, ctx.Err()
		}
		MarkResponseCommitted(c)
		if stream {
			_, _ = c.Writer.WriteString("event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"code\":\"basispoints_stream_incomplete\",\"message\":\"Upstream stream ended before completion\"}}}\n\n")
			c.Writer.Flush()
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
