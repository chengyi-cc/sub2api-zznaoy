package service

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/service/basispoints"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
)

// Connectivity and candy tests must use the account's selected protocol too.
// Probe failures never change account scheduling or fall back to native Codex.
func (s *AccountTestService) testExcelBPSConnection(c *gin.Context, account *Account, model, prompt, mode string) error {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("X-Accel-Buffering", "no")
	if isOpenAIImageModel(model) {
		return s.sendErrorAndEnd(c, "Excel BPS does not support image generation models")
	}
	ctx := c.Request.Context()
	payload := createOpenAITestPayload(model, true)
	if strings.TrimSpace(prompt) != "" {
		payload["input"] = prompt
	}
	applyCandyTestPayload(c, payload)
	if mode == AccountTestModeCompact {
		payload["input"] = []any{map[string]any{"role": "user", "content": "Summarize this conversation."}, map[string]any{"type": "compaction_trigger"}}
		payload["tool_choice"] = "none"
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return s.sendErrorAndEnd(c, "Cannot encode Excel BPS probe")
	}
	body, bridge, err := basispoints.Prepare(raw, fmt.Sprintf("probe:%d", account.ID), nil)
	if err != nil {
		return s.sendErrorAndEnd(c, err.Error())
	}
	// Production uses the shared token provider, including refresh. The fallback
	// supports standalone test-service construction without a gateway dependency.
	token := account.GetOpenAIAccessToken()
	if s.openaiGatewayService != nil {
		token, _, err = s.openaiGatewayService.GetAccessToken(ctx, account)
	}
	if err != nil || token == "" {
		return s.sendErrorAndEnd(c, "Excel BPS OAuth credential is unavailable")
	}
	accountID := excelBPSAccountID(account, token)
	if accountID == "" {
		return s.sendErrorAndEnd(c, "Excel BPS requires chatgpt_account_id")
	}
	req, err := newExcelBPSRequest(WithHTTPUpstreamRedirectsDisabled(WithHTTPUpstreamProfile(ctx, HTTPUpstreamProfileLongStream)), body, token, accountID)
	if err != nil {
		return s.sendErrorAndEnd(c, "Cannot create Excel BPS probe")
	}
	proxyURL := ""
	if account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}
	s.sendEvent(c, TestEvent{Type: "test_start", Model: model})
	resp, err := s.httpUpstream.Do(req, proxyURL, account.ID, account.Concurrency)
	if err != nil {
		return s.sendErrorAndEnd(c, "Excel BPS connection failed")
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return s.sendErrorAndEnd(c, fmt.Sprintf("Excel BPS returned HTTP %d; no fallback was attempted", resp.StatusCode))
	}
	pump := newOpenAISSEReadPump(bridge.Stream(resp.Body), 16<<20)
	defer pump.Close()
	heartbeat := time.NewTicker(15 * time.Second)
	defer heartbeat.Stop()
	sawText := false
	for pump.Next(ctx, 0, heartbeat.C, func() { _, _ = c.Writer.WriteString(": keepalive\n\n"); c.Writer.Flush() }) {
		if !strings.HasPrefix(pump.Text(), "data: ") {
			continue
		}
		event := gjson.Parse(strings.TrimPrefix(pump.Text(), "data: "))
		switch event.Get("type").String() {
		case "response.output_text.delta":
			sawText = true
			s.sendEvent(c, TestEvent{Type: "content", Text: event.Get("delta").String()})
		case "response.completed":
			if mode == AccountTestModeCompact && !event.Get(`response.output.#(type=="compaction")`).Exists() {
				return s.sendErrorAndEnd(c, "Excel BPS completed without a compaction item")
			}
			if !sawText {
				for _, item := range event.Get("response.output").Array() {
					for _, content := range item.Get("content").Array() {
						if content.Get("type").String() == "output_text" {
							s.sendEvent(c, TestEvent{Type: "content", Text: content.Get("text").String()})
						}
					}
				}
			}
			markCandyCompletion(c, true)
			s.sendEvent(c, TestEvent{Type: "test_complete", Success: true})
			return nil
		case "response.failed", "response.incomplete", "error":
			return s.sendErrorAndEnd(c, "Excel BPS did not complete the probe")
		}
	}
	return s.sendErrorAndEnd(c, "Excel BPS stream ended before completion")
}
