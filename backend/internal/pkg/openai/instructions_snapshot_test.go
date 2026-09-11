package openai

import (
	"crypto/sha256"
	"fmt"
	"strings"
	"testing"
)

// These snapshots were compared with the public rust-v0.153.4 sources:
// https://github.com/openai/codex/blob/rust-v0.153.4/codex-rs/models-manager/models.json
// https://github.com/openai/codex/blob/rust-v0.153.4/codex-rs/models-manager/prompt.md
// They are public source snapshots, not copies of a private traffic capture.
// Hashes lock prompt content after CRLF and outer-whitespace normalization.
func TestCodexOfficialPromptSnapshots(t *testing.T) {
	normalize := func(s string) string {
		return strings.TrimSpace(strings.ReplaceAll(s, "\r\n", "\n"))
	}
	for _, tc := range []struct {
		name, prompt, sha256 string
		models               []string
	}{
		{"gpt-5.2", instructionsGPT52, "e1aca575a0fa0b0c9e8a72314d4be6fd2a7547df5f92a19a3e33658ace0f79c3", []string{"gpt-5.2"}},
		{"gpt-5.4", instructionsGPT54Official, "46223554f4c456936ec2294667e368e6c4e592f3271364fa78fed662b8349291", []string{"gpt-5.4"}},
		{"gpt-5.5", instructionsGPT55Official, "ca4db22940fe7dee15df40285ea01b5710b6e7b8079ad6ba746c98a3e8771336", []string{"gpt-5.5"}},
		{"gpt-5.6-code-mode", instructionsGPT56CodeMode, "822b92294a217f46c8f9794589faee3b5ba6fbd2d61ac6c90168042829b8cf03", []string{"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}},
		{"gpt-6-astra", instructionsGPT6AstraOfficial, "be213cc3a9566255f6d43f61c54cbc33cafbc3461680ad6513afa0850050c7d6", []string{"gpt-6-astra"}},
		{"codex-auto-review", instructionsCodexAutoReview, "40a1232c8bd01a87dc2283e5ae3c75f2b054dc2a12cf04e5a279c26e5c541b9b", []string{"codex-auto-review"}},
		{"fallback", instructionsFallback01534, "ebfcbdce4a6c353e85d6cde37e508b89c77a902bd27771c91caba5fd494bdb83", []string{"gpt-5.6", "gpt-6", "gpt-5.3-codex", "gpt-5.3-codex-spark"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prompt := normalize(tc.prompt)
			if got := fmt.Sprintf("%x", sha256.Sum256([]byte(prompt))); got != tc.sha256 {
				t.Errorf("fixed-version prompt changed: sha256 = %s, want %s", got, tc.sha256)
			}
			for _, model := range tc.models {
				if normalize(CodexBaseInstructionsForModel(model)) != prompt {
					t.Errorf("model %q selected the wrong snapshot", model)
				}
			}
		})
	}
}
