package openai

import (
	"strings"
	"testing"
)

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// CodexBaseInstructionsForModel 应按模型返回对应的真实 Codex base prompt。
func TestCodexBaseInstructionsForModel(t *testing.T) {
	cases := []struct {
		model    string
		wantHead string
	}{
		{"gpt-6-astra", "You are Codex, an agent based on GPT-6"},
		{"gpt-6", "You are a coding agent running in the Codex CLI"},
		{"gpt-5.6", "You are a coding agent running in the Codex CLI"},
		{"openai/gpt-6-astra", "You are Codex, an agent based on GPT-6"},
		{"OPENAI/GPT-6_ASTRA", "You are Codex, an agent based on GPT-6"},
		{"gpt-6-astra-2026-09-01", "You are Codex, an agent based on GPT-6"},
		{"gpt-5-codex", "You are Codex, based on GPT-5"},
		{"gpt-5.3-codex", "You are a coding agent running in the Codex CLI"},
		{"gpt-5.3-codex-spark", "You are a coding agent running in the Codex CLI"},
		{"gpt-5.1-codex-max", "You are Codex, based on GPT-5"},
		{"gpt-5.2-codex", "You are Codex, based on GPT-5"},
		{"gpt-5.5", "You are Codex, a coding agent based on GPT-5"},
		{" GPT-5.5 ", "You are Codex, a coding agent based on GPT-5"},
		{"gpt-5.2", "You are GPT-5.2 running in the Codex CLI"},
		{"gpt-5.1", "You are GPT-5.1 running in the Codex CLI"},
		{"gpt-5", "You are Codex, a coding agent based on GPT-5"},   // 回退到最新（GPT-5.5）
		{"gpt-5.4", "You are Codex, a coding agent based on GPT-5"}, // 使用独立的 0.153.4 快照
		{"gpt-5.3", "You are Codex, a coding agent based on GPT-5"}, // 未单独维护 → 最新
		{"some-unknown-model", "You are Codex, a coding agent based on GPT-5"},
		{"", "You are Codex, a coding agent based on GPT-5"}, // 回退到最新
	}
	for _, c := range cases {
		got := strings.TrimSpace(CodexBaseInstructionsForModel(c.model))
		if got == "" {
			t.Errorf("model %q: got empty instructions", c.model)
			continue
		}
		if !strings.HasPrefix(got, c.wantHead) {
			t.Errorf("model %q: got prefix %q, want %q", c.model, firstLine(got), c.wantHead)
		}
	}
}

func TestCodexBareModelFallback(t *testing.T) {
	const fallbackHead = "You are a coding agent running in the Codex CLI"
	for _, model := range []string{"gpt-5.6", "gpt-6", "OPENAI/GPT-5.6", " openai/GPT_6 "} {
		t.Run(model, func(t *testing.T) {
			if CodexUsesInputDeveloperInstructions(model) {
				t.Fatal("bare model without catalog metadata must use top-level fallback instructions")
			}
			if got := CodexBaseInstructionsForModel(model); !strings.HasPrefix(got, fallbackHead) {
				t.Errorf("expected the official fallback prompt, got %q", firstLine(got))
			}
		})
	}
	for _, model := range []string{"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-6-astra", "codex-auto-review"} {
		t.Run(model, func(t *testing.T) {
			if !CodexUsesInputDeveloperInstructions(model) {
				t.Fatal("catalog code-mode model must use developer input instructions")
			}
			if strings.HasPrefix(CodexBaseInstructionsForModel(model), fallbackHead) {
				t.Fatal("catalog model must keep its dedicated prompt")
			}
		})
	}
}
