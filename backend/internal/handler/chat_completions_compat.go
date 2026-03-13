package handler

import (
	"bytes"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

func isClaudeModel(model string) bool {
	m := strings.ToLower(model)
	return strings.HasPrefix(m, "claude")
}

// ChatCompletionsRouter returns a handler that routes /v1/chat/completions requests
// to the correct backend based on the model name.
func ChatCompletionsRouter(openai *OpenAIGatewayHandler, claude *GatewayHandler) gin.HandlerFunc {
	return func(c *gin.Context) {
		body, err := io.ReadAll(c.Request.Body)
		if err != nil || len(body) == 0 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to read request body"})
			return
		}
		restoreBody(c, body)

		model := gjson.GetBytes(body, "model").String()
		if isClaudeModel(model) {
			handleClaudeChatCompletions(c, claude, body)
		} else {
			openai.ChatCompletions(c)
		}
	}
}

func handleClaudeChatCompletions(c *gin.Context, h *GatewayHandler, body []byte) {
	reqModel := gjson.GetBytes(body, "model").String()
	reqStream := gjson.GetBytes(body, "stream").Bool()

	anthropicBody := convertCCToAnthropicBody(body)
	restoreBody(c, anthropicBody)

	if reqStream {
		savedWriter := c.Writer
		c.Writer = newAnthropicSSEConverter(c.Writer, reqModel)
		defer func() { c.Writer = savedWriter }()
	}
	h.Messages(c)
}

func convertCCToAnthropicBody(body []byte) []byte {
	out := `{}`

	if m := gjson.GetBytes(body, "model"); m.Exists() {
		out, _ = sjson.Set(out, "model", m.Value())
	}
	if s := gjson.GetBytes(body, "stream"); s.Exists() {
		out, _ = sjson.Set(out, "stream", s.Value())
	}

	maxTokens := 8192
	if mt := gjson.GetBytes(body, "max_tokens"); mt.Exists() && mt.Type == gjson.Number {
		maxTokens = int(mt.Int())
	}
	if mt := gjson.GetBytes(body, "max_completion_tokens"); mt.Exists() && mt.Type == gjson.Number {
		maxTokens = int(mt.Int())
	}
	out, _ = sjson.Set(out, "max_tokens", maxTokens)

	if t := gjson.GetBytes(body, "temperature"); t.Exists() && t.Type == gjson.Number {
		out, _ = sjson.Set(out, "temperature", t.Value())
	}
	if tp := gjson.GetBytes(body, "top_p"); tp.Exists() && tp.Type == gjson.Number {
		out, _ = sjson.Set(out, "top_p", tp.Value())
	}

	var systemParts []string
	var messages []any
	gjson.GetBytes(body, "messages").ForEach(func(_, msg gjson.Result) bool {
		role := msg.Get("role").String()
		if role == "system" || role == "developer" {
			if s := messageTextContent(msg.Get("content")); s != "" {
				systemParts = append(systemParts, s)
			}
		} else {
			messages = append(messages, msg.Value())
		}
		return true
	})
	if len(systemParts) > 0 {
		out, _ = sjson.Set(out, "system", strings.Join(systemParts, "\n\n"))
	}
	out, _ = sjson.Set(out, "messages", messages)

	if tools := gjson.GetBytes(body, "tools"); tools.Exists() && tools.IsArray() {
		var converted []any
		tools.ForEach(func(_, tool gjson.Result) bool {
			if tool.Get("type").String() == "function" {
				fn := tool.Get("function")
				item := map[string]any{
					"name":        fn.Get("name").String(),
					"description": fn.Get("description").String(),
				}
				if params := fn.Get("parameters"); params.Exists() {
					item["input_schema"] = params.Value()
				}
				converted = append(converted, item)
			} else {
				converted = append(converted, tool.Value())
			}
			return true
		})
		out, _ = sjson.Set(out, "tools", converted)
	}
	if tc := gjson.GetBytes(body, "tool_choice"); tc.Exists() && !isUndefinedPlaceholder(tc) {
		out, _ = sjson.Set(out, "tool_choice", tc.Value())
	}

	return []byte(out)
}

// restoreBody resets c.Request.Body so downstream handlers can read it again.
func restoreBody(c *gin.Context, body []byte) {
	c.Request.Body = io.NopCloser(bytes.NewReader(body))
	c.Request.ContentLength = int64(len(body))
}

// isUndefinedPlaceholder returns true for the serialized "[undefined]" string
// that some clients (e.g. Cherry Studio) emit for unset optional fields.
func isUndefinedPlaceholder(r gjson.Result) bool {
	return r.Type == gjson.String && r.String() == "[undefined]"
}

// messageTextContent extracts plain text from a content field, handling both
// string and array-of-parts forms. Only text-type parts are included.
func messageTextContent(content gjson.Result) string {
	if content.IsArray() {
		parts := make([]string, 0, 2)
		content.ForEach(func(_, item gjson.Result) bool {
			switch item.Get("type").String() {
			case "text", "input_text", "output_text":
				if text := strings.TrimSpace(item.Get("text").String()); text != "" {
					parts = append(parts, text)
				}
			}
			return true
		})
		return strings.Join(parts, "\n\n")
	}
	return content.String()
}
