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
	if mt := gjson.GetBytes(body, "max_tokens"); mt.Exists() {
		maxTokens = int(mt.Int())
	}
	if mt := gjson.GetBytes(body, "max_completion_tokens"); mt.Exists() {
		maxTokens = int(mt.Int())
	}
	out, _ = sjson.Set(out, "max_tokens", maxTokens)

	if t := gjson.GetBytes(body, "temperature"); t.Exists() {
		out, _ = sjson.Set(out, "temperature", t.Value())
	}
	if tp := gjson.GetBytes(body, "top_p"); tp.Exists() {
		out, _ = sjson.Set(out, "top_p", tp.Value())
	}

	var systemParts []string
	var messages []any
	gjson.GetBytes(body, "messages").ForEach(func(_, msg gjson.Result) bool {
		role := msg.Get("role").String()
		if role == "system" || role == "developer" {
			if s := msg.Get("content").String(); s != "" {
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
	if tc := gjson.GetBytes(body, "tool_choice"); tc.Exists() {
		out, _ = sjson.Set(out, "tool_choice", tc.Value())
	}

	return []byte(out)
}

// ChatCompletions accepts requests on /v1/chat/completions and delegates to the
// Responses handler. Cursor (and similar clients) may send OpenAI Responses-format
// payloads to this legacy endpoint; we detect that case and pass through directly.
// For genuine Chat Completions payloads we do a best-effort conversion to the
// Responses API schema before forwarding.
func (h *OpenAIGatewayHandler) ChatCompletions(c *gin.Context) {
	savedWriter := c.Writer
	defer func() { c.Writer = savedWriter }()

	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Failed to read request body")
		return
	}
	if len(body) == 0 {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Request body is empty")
		return
	}
	if !gjson.ValidBytes(body) {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Invalid JSON")
		return
	}

	reqModel := gjson.GetBytes(body, "model").String()
	reqStream := gjson.GetBytes(body, "stream").Bool()

	if !isChatCompletionsFormat(body) {
		body = stripChatCompletionsOnlyFields(body)
		restoreBody(c, body)
		if reqStream {
			c.Writer = newSSEResponseConverter(c.Writer, reqModel)
		}
		h.Responses(c)
		return
	}

	converted, err := convertChatCompletionsToResponsesBody(body)
	if err != nil {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Failed to convert chat/completions request to responses format")
		return
	}

	restoreBody(c, converted)
	if reqStream {
		c.Writer = newSSEResponseConverter(c.Writer, reqModel)
	}
	h.Responses(c)
}

// chatCompletionsOnlyFields lists request fields that exist in the Chat
// Completions API but are not accepted by the Responses API. When a client
// sends a Responses-format body to /v1/chat/completions with these fields
// mixed in, we strip them before forwarding.
var chatCompletionsOnlyFields = []string{
	"stream_options",
	"metadata",
	"logprobs",
	"top_logprobs",
	"n",
	"seed",
	"stop",
	"logit_bias",
	"service_tier",
	"user",
	"response_format",
}

// stripChatCompletionsOnlyFields removes Chat Completions-specific parameters
// from a Responses-format body to prevent upstream 400 errors.
func stripChatCompletionsOnlyFields(body []byte) []byte {
	for _, key := range chatCompletionsOnlyFields {
		if gjson.GetBytes(body, key).Exists() {
			body, _ = sjson.DeleteBytes(body, key)
		}
	}
	return body
}

// isChatCompletionsFormat returns true when the payload looks like a standard
// OpenAI Chat Completions request (has "messages" array).
func isChatCompletionsFormat(body []byte) bool {
	return gjson.GetBytes(body, "messages").Exists()
}

// restoreBody resets c.Request.Body so downstream handlers can read it again.
func restoreBody(c *gin.Context, body []byte) {
	c.Request.Body = io.NopCloser(bytes.NewReader(body))
	c.Request.ContentLength = int64(len(body))
}

// convertChatCompletionsToResponsesBody converts an OpenAI Chat Completions
// request body into the Responses API format expected by the upstream handler.
//
// Mapping:
//
//	messages[role=system]         → instructions  (first system message)
//	messages[role=user/assistant] → input[]
//	messages[role=tool]           → input[] (function_call_output)
//	assistant tool_calls          → input[] (function_call items)
//	model / stream / temperature / top_p / max_tokens / stop → passthrough / rename
//	tools (function definitions)  → tools (flattened format)
func convertChatCompletionsToResponsesBody(body []byte) ([]byte, error) {
	out := `{}`
	var err error

	if m := gjson.GetBytes(body, "model"); m.Exists() {
		out, _ = sjson.Set(out, "model", m.Value())
	}
	if s := gjson.GetBytes(body, "stream"); s.Exists() {
		out, _ = sjson.Set(out, "stream", s.Value())
	}
	if t := gjson.GetBytes(body, "temperature"); t.Exists() {
		out, _ = sjson.Set(out, "temperature", t.Value())
	}
	if tp := gjson.GetBytes(body, "top_p"); tp.Exists() {
		out, _ = sjson.Set(out, "top_p", tp.Value())
	}
	if mt := gjson.GetBytes(body, "max_tokens"); mt.Exists() {
		out, _ = sjson.Set(out, "max_output_tokens", mt.Value())
	}
	if mt := gjson.GetBytes(body, "max_completion_tokens"); mt.Exists() {
		out, _ = sjson.Set(out, "max_output_tokens", mt.Value())
	}

	var input []any
	messages := gjson.GetBytes(body, "messages")
	messages.ForEach(func(_, msg gjson.Result) bool {
		role := msg.Get("role").String()
		switch role {
		case "system", "developer":
			// First system/developer message becomes instructions.
			if !gjson.Valid(gjson.Get(out, "instructions").Raw) || gjson.Get(out, "instructions").String() == "" {
				out, _ = sjson.Set(out, "instructions", msg.Get("content").String())
			}
		case "user":
			input = append(input, map[string]any{
				"role":    "user",
				"content": messageContent(msg),
			})
		case "assistant":
			if toolCalls := msg.Get("tool_calls"); toolCalls.Exists() && toolCalls.IsArray() {
				toolCalls.ForEach(func(_, tc gjson.Result) bool {
					item := map[string]any{
						"type":      "function_call",
						"call_id":   tc.Get("id").String(),
						"name":      tc.Get("function.name").String(),
						"arguments": tc.Get("function.arguments").String(),
					}
					input = append(input, item)
					return true
				})
			} else {
				input = append(input, map[string]any{
					"role":    "assistant",
					"content": messageContent(msg),
				})
			}
		case "tool":
			input = append(input, map[string]any{
				"type":    "function_call_output",
				"call_id": msg.Get("tool_call_id").String(),
				"output":  msg.Get("content").String(),
			})
		default:
			input = append(input, map[string]any{
				"role":    role,
				"content": messageContent(msg),
			})
		}
		return true
	})

	out, err = sjson.Set(out, "input", input)
	if err != nil {
		return nil, err
	}

	if tools := gjson.GetBytes(body, "tools"); tools.Exists() && tools.IsArray() {
		var converted []any
		tools.ForEach(func(_, tool gjson.Result) bool {
			if tool.Get("type").String() == "function" {
				fn := tool.Get("function")
				item := map[string]any{
					"type":        "function",
					"name":        fn.Get("name").String(),
					"description": fn.Get("description").String(),
				}
				if params := fn.Get("parameters"); params.Exists() {
					item["parameters"] = params.Value()
				}
				if strict := fn.Get("strict"); strict.Exists() {
					item["strict"] = strict.Bool()
				}
				converted = append(converted, item)
			} else {
				converted = append(converted, tool.Value())
			}
			return true
		})
		out, _ = sjson.Set(out, "tools", converted)
	}

	if tc := gjson.GetBytes(body, "tool_choice"); tc.Exists() {
		out, _ = sjson.Set(out, "tool_choice", tc.Value())
	}

	return []byte(out), nil
}

// messageContent extracts content from a message, handling both plain string
// content and array-of-parts content.
func messageContent(msg gjson.Result) any {
	c := msg.Get("content")
	if c.IsArray() {
		return c.Value()
	}
	return c.String()
}
