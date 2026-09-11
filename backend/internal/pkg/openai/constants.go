// Package openai provides helpers and types for OpenAI API integration.
package openai

import (
	_ "embed"
	"strings"
)

// Model represents an OpenAI model
type Model struct {
	ID          string `json:"id"`
	Object      string `json:"object"`
	Created     int64  `json:"created"`
	OwnedBy     string `json:"owned_by"`
	Type        string `json:"type"`
	DisplayName string `json:"display_name"`
}

// DefaultModels OpenAI models list
var DefaultModels = []Model{
	{ID: "gpt-5.6-sol", Object: "model", Created: 1780876800, OwnedBy: "openai", Type: "model", DisplayName: "GPT-5.6 Sol"},
	{ID: "gpt-6", Object: "model", Created: 1788480000, OwnedBy: "openai", Type: "model", DisplayName: "GPT-6 (Astra)"},
	{ID: "gpt-5.6", Object: "model", Created: 1780876800, OwnedBy: "openai", Type: "model", DisplayName: "GPT-5.6 (Sol)"},
	{ID: "gpt-5.6-terra", Object: "model", Created: 1780876800, OwnedBy: "openai", Type: "model", DisplayName: "GPT-5.6 Terra"},
	{ID: "gpt-5.6-luna", Object: "model", Created: 1780876800, OwnedBy: "openai", Type: "model", DisplayName: "GPT-5.6 Luna"},
	{ID: "gpt-6-astra", Object: "model", Created: 1788480000, OwnedBy: "openai", Type: "model", DisplayName: "GPT-6 Astra"},
	{ID: "gpt-5.5", Object: "model", Created: 1776873600, OwnedBy: "openai", Type: "model", DisplayName: "GPT-5.5"},
	{ID: "gpt-5.4", Object: "model", Created: 1738368000, OwnedBy: "openai", Type: "model", DisplayName: "GPT-5.4"},
	{ID: "gpt-5.4-mini", Object: "model", Created: 1738368000, OwnedBy: "openai", Type: "model", DisplayName: "GPT-5.4 Mini"},
	{ID: "gpt-5.3-codex-spark", Object: "model", Created: 1735689600, OwnedBy: "openai", Type: "model", DisplayName: "GPT-5.3 Codex Spark"},
	{ID: "codex-auto-review", Object: "model", Created: 1776902400, OwnedBy: "openai", Type: "model", DisplayName: "Codex Auto Review"},
	{ID: "gpt-5.2", Object: "model", Created: 1733875200, OwnedBy: "openai", Type: "model", DisplayName: "GPT-5.2"},
	{ID: "gpt-image-1", Object: "model", Created: 1733875200, OwnedBy: "openai", Type: "model", DisplayName: "GPT Image 1"},
	{ID: "gpt-image-1.5", Object: "model", Created: 1735689600, OwnedBy: "openai", Type: "model", DisplayName: "GPT Image 1.5"},
	{ID: "gpt-image-2", Object: "model", Created: 1738368000, OwnedBy: "openai", Type: "model", DisplayName: "GPT Image 2"},
	{ID: "gpt-image-2.5-flare", Object: "model", Created: 1788825600, OwnedBy: "openai", Type: "model", DisplayName: "GPT Image 2.5 Flare"},
	{ID: "gpt-image-2.5-sunburst", Object: "model", Created: 1788825600, OwnedBy: "openai", Type: "model", DisplayName: "GPT Image 2.5 Sunburst"},
}

// DefaultModelIDs returns the default model ID list
func DefaultModelIDs() []string {
	ids := make([]string, len(DefaultModels))
	for i, m := range DefaultModels {
		ids[i] = m.ID
	}
	return ids
}

// DefaultTestModel default model for testing OpenAI accounts
const DefaultTestModel = "gpt-5.4"

// CodexUsageProbeModel is the model used for OAuth Codex usage probes.
const CodexUsageProbeModel = "codex-auto-review"

// DefaultInstructions default instructions for non-Codex CLI requests.
// 内容为真实 Codex CLI 的 GPT-5-Codex base prompt（codex 系模型默认）。
//
//go:embed instructions.txt
var DefaultInstructions string

// instructionsGPT51 / instructionsGPT52 / instructionsGPT55 / instructionsGPT6Astra
// 为对应非 codex 模型的真实 Codex 编码 agent base prompt，用于模型感知的 instructions 选择。
// GPT-5.5 同时作为 GPT-5 系列的 fallback（覆盖 5.3 / 5.4 等未单独维护 prompt 的版本）。
//
//go:embed instructions_gpt5_1.txt
var instructionsGPT51 string

//go:embed instructions_gpt5_2.txt
var instructionsGPT52 string

//go:embed instructions_gpt5_5.txt
var instructionsGPT55 string

//go:embed instructions_gpt5_4_official.txt
var instructionsGPT54Official string

//go:embed instructions_gpt5_5_official.txt
var instructionsGPT55Official string

//go:embed instructions_gpt5_6_code_mode.txt
var instructionsGPT56CodeMode string

//go:embed instructions_codex_auto_review.txt
var instructionsCodexAutoReview string

// Source: openai/codex codex-rs/models-manager/models.json at 121f91fd5d9d.
//
//go:embed instructions_gpt6_astra.txt
var instructionsGPT6Astra string

//go:embed instructions_gpt6_astra_official.txt
var instructionsGPT6AstraOfficial string

// Source: openai/codex rust-v0.153.4, codex-rs/models-manager/prompt.md.
// The catalog has no entries for bare gpt-5.6, gpt-6, or gpt-5.3-codex
// (including Spark); model_info.rs uses this fallback text for these models.
//
//go:embed instructions_fallback_0_153_4.txt
var instructionsFallback01534 string

// latestCodexInstructions 返回当前已知最新版本的 Codex base instructions，
// 当前为 GPT-5.5；若 5.5 prompt 意外为空则回退到 DefaultInstructions 保证非空。
func latestCodexInstructions() string {
	if v := strings.TrimSpace(instructionsGPT55Official); v != "" {
		return instructionsGPT55Official
	}
	if v := strings.TrimSpace(instructionsGPT55); v != "" {
		return instructionsGPT55
	}
	return DefaultInstructions
}

// CanonicalizeOpenAIModelAliasSpelling normalizes provider prefixes, case,
// separators, and known compact spellings used by OpenAI model aliases.
func CanonicalizeOpenAIModelAliasSpelling(model string) string {
	model = strings.TrimSpace(model)
	if slash := strings.LastIndexByte(model, '/'); slash >= 0 {
		model = strings.TrimSpace(model[slash+1:])
	}
	model = strings.ToLower(model)
	if model == "" {
		return ""
	}

	normalized := strings.ReplaceAll(model, "_", "-")
	normalized = strings.Join(strings.Fields(normalized), "-")
	for strings.Contains(normalized, "--") {
		normalized = strings.ReplaceAll(normalized, "--", "-")
	}

	if strings.HasPrefix(normalized, "gpt5") {
		normalized = "gpt-5" + strings.TrimPrefix(normalized, "gpt5")
	}
	if !strings.HasPrefix(normalized, "gpt-") && !strings.Contains(normalized, "codex") {
		return ""
	}

	replacements := []struct {
		from string
		to   string
	}{
		{"gpt-5.4mini", "gpt-5.4-mini"},
		{"gpt-5.4nano", "gpt-5.4-nano"},
		{"gpt-5.3-codexspark", "gpt-5.3-codex-spark"},
		{"gpt-5.3codexspark", "gpt-5.3-codex-spark"},
		{"gpt-5.3codex", "gpt-5.3-codex"},
	}
	for _, replacement := range replacements {
		normalized = strings.ReplaceAll(normalized, replacement.from, replacement.to)
	}
	return normalized
}

// CodexBaseInstructionsForModel 按模型返回最匹配的真实 Codex base instructions：
//   - 裸 gpt-5.6 / gpt-6 → 官方 0.153.4 无模型元数据时的默认 prompt
//   - gpt-5.3-codex / spark → 同一官方回退 prompt
//   - gpt-6-astra（含供应商前缀与日期变体）→ GPT-6 Astra prompt
//   - gpt-5.6-sol/terra/luna / codex-auto-review → 对应代码模式 prompt
//   - 含 "codex" 的模型（gpt-5-codex / gpt-5.x-codex / codex-max / spark 等）→ GPT-5-Codex prompt
//   - gpt-5.4 系非 codex 模型 → GPT-5.4 prompt
//   - gpt-5.5 系非 codex 模型 → GPT-5.5 prompt
//   - gpt-5.2 系非 codex 模型 → GPT-5.2 prompt
//   - gpt-5.1 系非 codex 模型 → GPT-5.1 prompt
//   - 其它（含 gpt-5.3 / 裸 gpt-5 / 未知模型）→ 保留原有回退规则（当前 GPT-5.5）
//
// 任一专用 prompt 意外为空时回退链最终落到 DefaultInstructions，保证返回非空。
func CodexBaseInstructionsForModel(model string) string {
	canonical := CanonicalizeOpenAIModelAliasSpelling(model)
	switch {
	case canonical == "gpt-5.6" || canonical == "gpt-6" || canonical == "gpt-5.3-codex" || strings.HasPrefix(canonical, "gpt-5.3-codex-"):
		if v := strings.TrimSpace(instructionsFallback01534); v != "" {
			return instructionsFallback01534
		}
	case canonical == "codex-auto-review":
		if v := strings.TrimSpace(instructionsCodexAutoReview); v != "" {
			return instructionsCodexAutoReview
		}
	case strings.HasPrefix(canonical, "gpt-5.6-sol"), strings.HasPrefix(canonical, "gpt-5.6-terra"), strings.HasPrefix(canonical, "gpt-5.6-luna"):
		if v := strings.TrimSpace(instructionsGPT56CodeMode); v != "" {
			return instructionsGPT56CodeMode
		}
	case strings.HasPrefix(canonical, "gpt-5.4"):
		if v := strings.TrimSpace(instructionsGPT54Official); v != "" {
			return instructionsGPT54Official
		}
	case canonical == "gpt-6-astra" || strings.HasPrefix(canonical, "gpt-6-astra-"):
		if v := strings.TrimSpace(instructionsGPT6AstraOfficial); v != "" {
			return instructionsGPT6AstraOfficial
		}
		if v := strings.TrimSpace(instructionsGPT6Astra); v != "" {
			return instructionsGPT6Astra
		}
	case strings.Contains(canonical, "codex"):
		return DefaultInstructions
	case strings.HasPrefix(canonical, "gpt-5.5"):
		return latestCodexInstructions()
	case strings.HasPrefix(canonical, "gpt-5.2"):
		if v := strings.TrimSpace(instructionsGPT52); v != "" {
			return instructionsGPT52
		}
	case strings.HasPrefix(canonical, "gpt-5.1"):
		if v := strings.TrimSpace(instructionsGPT51); v != "" {
			return instructionsGPT51
		}
	}
	return latestCodexInstructions()
}

// CodexUsesInputDeveloperInstructions reports models whose official catalog
// uses code-mode input (代码模式输入) instead of top-level instructions.
func CodexUsesInputDeveloperInstructions(model string) bool {
	canonical := CanonicalizeOpenAIModelAliasSpelling(model)
	return canonical == "gpt-6-astra" ||
		strings.HasPrefix(canonical, "gpt-6-astra-") ||
		strings.HasPrefix(canonical, "gpt-5.6-sol") ||
		strings.HasPrefix(canonical, "gpt-5.6-terra") ||
		strings.HasPrefix(canonical, "gpt-5.6-luna") ||
		canonical == "codex-auto-review"
}
