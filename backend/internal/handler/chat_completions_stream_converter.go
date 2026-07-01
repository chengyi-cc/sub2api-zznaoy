package handler

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
)

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// ---------------------------------------------------------------------------
// Anthropic SSE → Chat Completions converter
// ---------------------------------------------------------------------------

type anthropicSSEConverter struct {
	gin.ResponseWriter
	model   string
	chatID  string
	created int64

	sentRole    bool
	sentDone    bool
	toolIdx     int
	passthrough bool
	checkedCT   bool
	lineBuf     bytes.Buffer
}

func newAnthropicSSEConverter(w gin.ResponseWriter, model string) *anthropicSSEConverter {
	return &anthropicSSEConverter{
		ResponseWriter: w,
		model:          model,
		chatID:         fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		created:        time.Now().Unix(),
	}
}

func (conv *anthropicSSEConverter) Write(data []byte) (int, error) {
	if conv.anthropicShouldPassthrough() {
		return conv.ResponseWriter.Write(data)
	}
	log.Printf("[ANTHROPIC-CONV] incoming write len=%d data=%s", len(data), truncate(string(data), 300))
	_, _ = conv.lineBuf.Write(data)
	if err := conv.anthropicProcessLines(); err != nil {
		return 0, err
	}
	return len(data), nil
}

func (conv *anthropicSSEConverter) WriteString(s string) (int, error) {
	if conv.anthropicShouldPassthrough() {
		return conv.ResponseWriter.WriteString(s)
	}
	_, _ = conv.lineBuf.WriteString(s)
	if err := conv.anthropicProcessLines(); err != nil {
		return 0, err
	}
	return len(s), nil
}

func (conv *anthropicSSEConverter) Flush() {
	if f, ok := conv.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (conv *anthropicSSEConverter) anthropicShouldPassthrough() bool {
	if conv.checkedCT {
		return conv.passthrough
	}
	conv.checkedCT = true
	ct := conv.ResponseWriter.Header().Get("Content-Type")
	if ct != "" && !strings.Contains(ct, "text/event-stream") {
		conv.passthrough = true
	}
	return conv.passthrough
}

func (conv *anthropicSSEConverter) anthropicProcessLines() error {
	for {
		buf := conv.lineBuf.Bytes()
		idx := bytes.IndexByte(buf, '\n')
		if idx < 0 {
			break
		}
		line := string(buf[:idx])
		conv.lineBuf.Next(idx + 1)
		line = strings.TrimRight(line, "\r")
		if err := conv.anthropicProcessLine(line); err != nil {
			return err
		}
	}
	return nil
}

func (conv *anthropicSSEConverter) anthropicProcessLine(line string) error {
	if strings.HasPrefix(line, "event:") || strings.HasPrefix(line, ":") || line == "" {
		return nil
	}
	if !strings.HasPrefix(line, "data:") {
		return nil
	}

	data := strings.TrimPrefix(line, "data:")
	data = strings.TrimSpace(data)
	if data == "[DONE]" || !gjson.Valid(data) {
		return nil
	}

	eventType := gjson.Get(data, "type").String()

	switch eventType {
	case "message_start":
		if id := gjson.Get(data, "message.id").String(); id != "" {
			conv.chatID = "chatcmpl-" + strings.TrimPrefix(id, "msg_")
		}
		if m := gjson.Get(data, "message.model").String(); m != "" {
			conv.model = m
		}
		return conv.anthropicEnsureRoleChunk()

	case "content_block_start":
		blockType := gjson.Get(data, "content_block.type").String()
		if blockType == "tool_use" {
			callID := gjson.Get(data, "content_block.id").String()
			name := gjson.Get(data, "content_block.name").String()
			return conv.anthropicWriteToolCallStartChunk(callID, name)
		}

	case "content_block_delta":
		deltaType := gjson.Get(data, "delta.type").String()
		switch deltaType {
		case "text_delta":
			text := gjson.Get(data, "delta.text").String()
			if text != "" {
				return conv.anthropicWriteContentChunk(text)
			}
		case "input_json_delta":
			json := gjson.Get(data, "delta.partial_json").String()
			if json != "" {
				return conv.anthropicWriteToolCallArgChunk(json)
			}
		}

	case "message_delta":
		stopReason := gjson.Get(data, "delta.stop_reason").String()
		return conv.anthropicWriteFinishChunk(stopReason)

	case "message_stop":
		if !conv.sentDone {
			conv.sentDone = true
			return conv.anthropicWriteDone()
		}
	}

	return nil
}

func (conv *anthropicSSEConverter) anthropicMakeChunk(delta map[string]any, finishReason *string) ([]byte, error) {
	chunk := map[string]any{
		"id":      conv.chatID,
		"object":  "chat.completion.chunk",
		"created": conv.created,
		"model":   conv.model,
		"choices": []map[string]any{{
			"index":         0,
			"delta":         delta,
			"finish_reason": finishReason,
		}},
	}
	return json.Marshal(chunk)
}

func (conv *anthropicSSEConverter) anthropicWriteChunkBytes(b []byte) error {
	var buf bytes.Buffer
	_, _ = buf.WriteString("data: ")
	_, _ = buf.Write(b)
	_, _ = buf.WriteString("\n\n")
	log.Printf("[ANTHROPIC-CONV] outgoing chunk=%s", truncate(buf.String(), 300))
	if _, err := conv.ResponseWriter.Write(buf.Bytes()); err != nil {
		return err
	}
	conv.Flush()
	return nil
}

func (conv *anthropicSSEConverter) anthropicEnsureRoleChunk() error {
	if conv.sentRole {
		return nil
	}
	conv.sentRole = true
	b, err := conv.anthropicMakeChunk(map[string]any{"role": "assistant"}, nil)
	if err != nil {
		return err
	}
	return conv.anthropicWriteChunkBytes(b)
}

func (conv *anthropicSSEConverter) anthropicWriteContentChunk(content string) error {
	if err := conv.anthropicEnsureRoleChunk(); err != nil {
		return err
	}
	b, err := conv.anthropicMakeChunk(map[string]any{"content": content}, nil)
	if err != nil {
		return err
	}
	return conv.anthropicWriteChunkBytes(b)
}

func (conv *anthropicSSEConverter) anthropicWriteToolCallStartChunk(callID, name string) error {
	if err := conv.anthropicEnsureRoleChunk(); err != nil {
		return err
	}
	idx := conv.toolIdx
	conv.toolIdx++
	b, err := conv.anthropicMakeChunk(map[string]any{
		"tool_calls": []map[string]any{{
			"index": idx,
			"id":    callID,
			"type":  "function",
			"function": map[string]any{
				"name":      name,
				"arguments": "",
			},
		}},
	}, nil)
	if err != nil {
		return err
	}
	return conv.anthropicWriteChunkBytes(b)
}

func (conv *anthropicSSEConverter) anthropicWriteToolCallArgChunk(args string) error {
	idx := conv.toolIdx - 1
	if idx < 0 {
		idx = 0
	}
	b, err := conv.anthropicMakeChunk(map[string]any{
		"tool_calls": []map[string]any{{
			"index": idx,
			"function": map[string]any{
				"arguments": args,
			},
		}},
	}, nil)
	if err != nil {
		return err
	}
	return conv.anthropicWriteChunkBytes(b)
}

func (conv *anthropicSSEConverter) anthropicWriteFinishChunk(stopReason string) error {
	reason := "stop"
	switch stopReason {
	case "tool_use":
		reason = "tool_calls"
	case "max_tokens":
		reason = "length"
	}
	b, err := conv.anthropicMakeChunk(map[string]any{}, &reason)
	if err != nil {
		return err
	}
	return conv.anthropicWriteChunkBytes(b)
}

func (conv *anthropicSSEConverter) anthropicWriteDone() error {
	if _, err := conv.ResponseWriter.Write([]byte("data: [DONE]\n\n")); err != nil {
		return err
	}
	conv.Flush()
	return nil
}
