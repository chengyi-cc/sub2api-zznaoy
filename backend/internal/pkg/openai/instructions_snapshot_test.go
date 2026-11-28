package openai

import (
	"crypto/sha256"
	"fmt"
	"strings"
	"testing"
)

// These hashes were computed from the public rust-v0.172.0 sources, not from
// the files in this package, so the test proves each bundled prompt is the
// official text and that every model selects the prompt official Codex uses:
// https://github.com/openai/codex/blob/rust-v0.172.0/codex-rs/models-manager/models.json
// https://github.com/openai/codex/blob/rust-v0.172.0/codex-rs/models-manager/prompt.md
// Hashes lock prompt content after CRLF and outer-whitespace normalization.
func TestCodexOfficialPromptSnapshots(t *testing.T) {
	normalize := func(s string) string {
		return strings.TrimSpace(strings.ReplaceAll(s, "\r\n", "\n"))
	}
	for _, tc := range []struct {
		name, sha256 string
		models       []string
	}{
		{"gpt-5.5", "3d7984a1671ad1c65f6bd7732f03b69eb96b616cd7fa7390eb5e5ca04d06cf49", []string{"gpt-5.5"}},
		// codex-auto-review shares the gpt-5.6-* template in the official catalog.
		{"gpt-5.6", "0972574f9d43b6c30321624b9ba5eec3a47db32251b7ba6d94807bf5295170e3", []string{"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "codex-auto-review"}},
		{"gpt-6-astra", "90d6721b2848a8462e1ca2bb5aac10fad29ab7707a669d982b504e98a332e9b1", []string{"gpt-6-astra", "gpt-6.1-terra"}},
		{"gpt-6-sol", "8fa2135c6f2f3369f6f8e507c63371b097503bd6d02c248d45cb87b0c5185ca9", []string{"gpt-6-sol"}},
		{"gpt-6-luna", "4a9dd6b80dd8acba9d82183141c9c1b0f7ac273558de414c6e2734e7b9fc2b29", []string{"gpt-6-luna"}},
		{"gpt-6.1-sol", "3c43dfaabb60f17def6b94fb672f4600b08b4f35ca4e27912595947329fe9da2", []string{"gpt-6.1-sol"}},
		{"gpt-6.2", "6c8bdad0f92784f908323ae3bcf7cb12c2fcd15fd7fbb36bc67d608da826165e", []string{"gpt-6.2-sol", "gpt-6.2-astra"}},
		// Models without a catalog entry fall back to prompt.md.
		{"prompt.md", "ebfcbdce4a6c353e85d6cde37e508b89c77a902bd27771c91caba5fd494bdb83", []string{"gpt-5.6", "gpt-6", "gpt-5.3-codex", "gpt-5.3-codex-spark"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, model := range tc.models {
				prompt := normalize(CodexBaseInstructionsForModel(model))
				if got := fmt.Sprintf("%x", sha256.Sum256([]byte(prompt))); got != tc.sha256 {
					t.Errorf("model %q: prompt sha256 = %s, want official rust-v0.172.0 %s", model, got, tc.sha256)
				}
			}
		})
	}
}
