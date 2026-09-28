package basispoints

import "testing"

func TestNativeRoutingEncryptedMessageHistory(t *testing.T) {
	for _, body := range []string{
		"{\"input\":[{\"type\":\"agent_message\",\"content\":[{\"type\":\"input_text\",\"text\":\"Payload:\"},{\"type\":\"encrypted_content\",\"encrypted_content\":\"opaque\"}]}]}",
		"{\"input\":[{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"encrypted_content\",\"encrypted_content\":\"opaque\"}]}]}",
		"{\"tool_choice\":\"none\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"encrypted_content\",\"encrypted_content\":\"opaque\"}]}]}",
	} {
		if got := NativeFallbackReason([]byte(body)); got != "encrypted_message_history" {
			t.Fatalf("got %q", got)
		}
	}
	for _, body := range []string{
		"{\"input\":[{\"type\":\"reasoning\",\"encrypted_content\":\"opaque\"}]}",
		"{\"input\":[{\"type\":\"compaction\",\"encrypted_content\":\"opaque\"}]}",
		"{\"input\":[{\"type\":\"agent_message\",\"content\":[{\"type\":\"input_text\",\"text\":\"plaintext result\"}]}]}",
		"{\"input\":[{\"type\":\"function_call\",\"arguments\":\"encrypted_content\"}]}",
	} {
		if got := NativeFallbackReason([]byte(body)); got != "" {
			t.Fatalf("unrelated history routed to native: %q", got)
		}
	}
}
