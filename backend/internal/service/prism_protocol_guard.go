package service

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	coderws "github.com/coder/websocket"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
)

func withPrismBrowserModelGuard(account *Account, hooks *OpenAIWSIngressHooks) *OpenAIWSIngressHooks {
	if !accountHasPrismBrowser(account) {
		return hooks
	}
	guarded := &OpenAIWSIngressHooks{}
	if hooks != nil {
		*guarded = *hooks
	}
	previous := guarded.MapRequestModel
	guarded.MapRequestModel = func(turn int, model string) (string, error) {
		mapped := model
		if previous != nil {
			value, err := previous(turn, model)
			if err != nil {
				return "", err
			}
			if strings.TrimSpace(value) != "" {
				mapped = value
			}
		}
		if account.IsPrismBrowserEnabledForModel(mapped) {
			return "", NewOpenAIWSClientCloseError(coderws.StatusPolicyViolation, "this model uses Prism; use HTTP /v1/responses", nil)
		}
		return mapped, nil
	}
	return guarded
}

// Unsupported entry points must not silently bypass an account's chosen route.
func rejectPrismCompatibility(c *gin.Context, account *Account, body []byte, dispatchModel string) error {
	model := resolveOpenAIForwardModel(account, gjson.GetBytes(body, "model").String(), dispatchModel)
	if !account.isPrismBrowserUpstreamModelEnabled(model) {
		return nil
	}
	MarkPrismBrowserAttempt(c, account.ID)
	err := errors.New("Prism currently requires HTTP /v1/responses; this compatibility endpoint is not supported")
	payload := gin.H{"error": gin.H{"type": "unsupported_prism_endpoint", "message": err.Error()}}
	committed := StopOpenAICompactSSEKeepaliveCommitted(c)
	if committed || c.Writer.Written() {
		prefix := ""
		if c.Request != nil && strings.HasSuffix(c.Request.URL.Path, "/messages") {
			prefix = "event: error\n"
			payload["type"] = "error"
		}
		raw, _ := json.Marshal(payload)
		_, _ = c.Writer.WriteString(prefix + "data: " + string(raw) + "\n\n")
		c.Writer.Flush()
	} else {
		c.JSON(http.StatusBadRequest, payload)
	}
	MarkResponseCommitted(c)
	return err
}
