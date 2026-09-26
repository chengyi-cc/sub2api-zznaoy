package errorarchive

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tidwall/gjson"
)

// RequestSummary inspects the whole decoded JSON, including history beyond the
// wire capture prefix. It contains structure, never message text, tool argument
// values, image payloads, URLs, encryption blobs, credentials or instructions.
// Detail samples are bounded separately from totals; this is not a replay body.
func RequestSummary(body []byte) json.RawMessage {
	result := map[string]any{"schema_version": 1, "decoded_bytes": len(body), "valid_json": gjson.ValidBytes(body)}
	if result["valid_json"] != true {
		raw, _ := json.Marshal(result)
		return raw
	}
	root := gjson.ParseBytes(body)
	clip := func(s string) string {
		r := []rune(s)
		if len(r) > 96 {
			return string(r[:96])
		}
		return s
	}
	result["model"] = clip(root.Get("model").String())
	result["stream"] = root.Get("stream").Bool()
	if v := root.Get("parallel_tool_calls"); v.Type == gjson.True || v.Type == gjson.False {
		result["parallel_tool_calls"] = v.Bool()
	}
	inputCount, imageCount, inlineCount, contentCount := 0, 0, 0, 0
	var encodedBytes int64
	types := map[string]int{}
	var samples []map[string]any
	detailsTruncated := false
	record := func(path string, value gjson.Result) {
		kind := clip(value.Get("type").String())
		if kind == "" {
			kind = "unknown"
		}
		if _, exists := types[kind]; !exists && len(types) >= 32 {
			kind = "other"
		}
		types[kind]++
		contentCount++
		if kind == "input_image" || kind == "image" {
			imageCount++
			url := value.Get("image_url").String()
			if strings.HasPrefix(url, "data:") {
				inlineCount++
				encodedBytes += int64(len(url))
			}
		}
		if len(samples) < 96 {
			samples = append(samples, map[string]any{"path": path, "type": kind})
		} else {
			detailsTruncated = true
		}
	}
	root.Get("input").ForEach(func(_, item gjson.Result) bool {
		i := inputCount
		inputCount++
		for _, field := range []string{"content", "output"} {
			parts := item.Get(field)
			if !parts.IsArray() {
				continue
			}
			index := 0
			parts.ForEach(func(_, part gjson.Result) bool {
				record(fmt.Sprintf("input[%d].%s[%d]", i, field, index), part)
				index++
				return true
			})
		}
		return true
	})
	if !root.Get("input").IsArray() {
		inputCount = 0
	}
	toolCount := 0
	var catalog []map[string]any
	appendTool := func(namespace string, entry gjson.Result) {
		toolCount++
		if len(catalog) < 128 {
			catalog = append(catalog, map[string]any{"namespace": clip(namespace), "name": clip(entry.Get("name").String()), "type": clip(entry.Get("type").String())})
		} else {
			detailsTruncated = true
		}
	}
	root.Get("tools").ForEach(func(_, entry gjson.Result) bool {
		if entry.Get("type").String() == "namespace" {
			entry.Get("tools").ForEach(func(_, child gjson.Result) bool { appendTool(entry.Get("name").String(), child); return true })
		} else {
			appendTool("", entry)
		}
		return true
	})
	result["input_items"], result["content_parts"], result["content_types"] = inputCount, contentCount, types
	result["images"], result["inline_images"], result["inline_image_encoded_bytes"] = imageCount, inlineCount, encodedBytes
	result["content_samples"], result["tool_count"], result["tool_catalog"] = samples, toolCount, catalog
	result["details_truncated"] = detailsTruncated
	raw, _ := json.Marshal(result)
	if len(raw) > 64<<10 {
		delete(result, "tool_catalog")
		delete(result, "content_samples")
		result["details_truncated"] = true
		raw, _ = json.Marshal(result)
	}
	return raw
}
