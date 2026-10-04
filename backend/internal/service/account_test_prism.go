package service

import (
	"encoding/json"
	"fmt"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"net/http"
	"strings"
)

func (s *AccountTestService) testPrismBrowserConnection(c *gin.Context, account *Account, modelID, prompt string) error {
	if s.openaiGatewayService == nil {
		return s.sendErrorAndEnd(c, "Prism gateway service is unavailable")
	}
	modelID = strings.TrimSpace(modelID)
	if modelID == "" {
		modelID = "gpt-5.6-sol"
	}
	modelID = account.GetMappedModel(modelID)
	if prompt == "" {
		prompt = "hi"
	}
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("X-Accel-Buffering", "no")
	s.sendEvent(c, TestEvent{Type: "test_start", Model: modelID})
	body, err := json.Marshal(map[string]any{"model": modelID, "input": prompt, "stream": false})
	if err != nil {
		return s.sendErrorAndEnd(c, "Invalid Prism test request")
	}
	response, _, status, err := s.openaiGatewayService.callPrismBrowser(c.Request.Context(), account, body)
	if err != nil {
		return s.sendErrorAndEnd(c, fmt.Sprintf("Prism adapter request failed: %s", err.Error()))
	}
	if status != http.StatusOK {
		return s.sendErrorAndEnd(c, prismBrowserAdapterErrorMessage(status, response))
	}
	if _, err := prismBrowserTerminal(response, modelID, false); err != nil {
		return s.sendErrorAndEnd(c, "Prism adapter returned no completed response")
	}
	answer := gjson.GetBytes(response, "output.0.content.0.text").String()
	if answer == "" {
		return s.sendErrorAndEnd(c, "Prism adapter returned no text")
	}
	s.sendEvent(c, TestEvent{Type: "content", Text: answer})
	s.sendEvent(c, TestEvent{Type: "test_complete", Success: true})
	return nil
}
