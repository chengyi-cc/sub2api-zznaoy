package basispoints

import (
	"strings"

	"github.com/tidwall/gjson"
)

// Route before sending anything: native hosted tools and unsupported content must not
// be silently dropped or emulated by a client tool on the Excel channel.
func NativeFallbackReason(body []byte) string {
	if !gjson.ValidBytes(body) {
		return ""
	}
	choice := gjson.GetBytes(body, "tool_choice")
	if choice.Exists() && choice.Type != gjson.Null && choice.String() != "auto" && choice.String() != "none" {
		return "tool_choice"
	}
	var inspectTools func(gjson.Result) bool
	inspectTools = func(tools gjson.Result) bool {
		for _, tool := range tools.Array() {
			kind := strings.TrimSpace(tool.Get("type").String())
			if isUnsupportedHostedTool(kind) {
				return true
			}
			if kind == "namespace" && inspectTools(tool.Get("tools")) {
				return true
			}
		}
		return false
	}
	if choice.String() != "none" && inspectTools(gjson.GetBytes(body, "tools")) {
		return "hosted_tools"
	}
	for _, item := range gjson.GetBytes(body, "input").Array() {
		// Collaboration payloads may contain the only copy of another agent's
		// result. BPS cannot lower these encrypted message parts to text. Select
		// the native route before translating, without dropping or guessing them.
		kind := item.Get("type").String()
		if kind == "agent_message" || kind == "message" || (kind == "" && item.Get("role").Exists()) {
			for _, part := range item.Get("content").Array() {
				if part.Get("type").String() == "encrypted_content" {
					return "encrypted_message_history"
				}
			}
		}
		if choice.String() != "none" && item.Get("type").String() == "additional_tools" && inspectTools(item.Get("tools")) {
			return "hosted_tools"
		}
		for _, field := range []string{"content", "output"} {
			for _, part := range item.Get(field).Array() {
				switch part.Get("type").String() {
				case "input_file", "input_audio", "audio":
					return "native_media"
				}
			}
		}
	}
	return ""
}
