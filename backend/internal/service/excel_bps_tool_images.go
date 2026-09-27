package service

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// File IDs and HTTPS tool image references belong in BPS message content, not
// function_call_output.output. Preserve text and association with each call.
// Run for both newly uploaded and already referenced images, after upload
// so synthetic messages cannot affect task/turn/cache identities in Prepare.
func normalizeExcelBPSToolOutputImages(wire []byte) ([]byte, error) {
	var input, pending []json.RawMessage
	changed := false
	for _, item := range gjson.GetBytes(wire, "input").Array() {
		raw := []byte(item.Raw)
		if item.Get("type").String() != "function_call_output" {
			// Keep parallel tool results together before their attachment messages.
			input = append(input, pending...)
			pending = nil
			input = append(input, raw)
			continue
		}
		var images []json.RawMessage
		for index, part := range item.Get("output").Array() {
			if part.Get("type").String() != "input_image" || (part.Get("file_id").String() == "" && !strings.HasPrefix(strings.ToLower(part.Get("image_url").String()), "https://")) {
				continue
			}
			images = append(images, json.RawMessage(part.Raw))
			marker := map[string]string{"type": "input_text", "text": fmt.Sprintf("[Tool output image %d for call_id %q is attached below.]", len(images), item.Get("call_id").String())}
			var err error
			raw, err = sjson.SetBytes(raw, fmt.Sprintf("output.%d", index), marker)
			if err != nil {
				return nil, fmt.Errorf("cannot normalize Excel BPS tool image output")
			}
		}
		if len(images) > 0 {
			label, _ := json.Marshal(map[string]string{"type": "input_text", "text": fmt.Sprintf("Images returned by tool call %q. These are tool output, not a new user request.", item.Get("call_id").String())})
			content := append([]json.RawMessage{label}, images...)
			attachment, err := json.Marshal(map[string]any{"type": "message", "role": "user", "content": content})
			if err != nil {
				return nil, fmt.Errorf("cannot encode Excel BPS tool image message")
			}
			pending = append(pending, attachment)
			changed = true
		}
		input = append(input, raw)
	}
	if !changed {
		return wire, nil
	}
	input = append(input, pending...)
	encoded, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("cannot encode Excel BPS image history")
	}
	updated, err := sjson.SetRawBytes(wire, "input", encoded)
	if err != nil {
		return nil, fmt.Errorf("cannot normalize Excel BPS image history")
	}
	return updated, nil
}
