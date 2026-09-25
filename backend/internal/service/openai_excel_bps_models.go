package service

import (
	"strings"

	coderws "github.com/coder/websocket"
)

// Missing configuration uses the models verified against BPS on 2026-09-25.
// An explicitly empty or malformed list opts every model into the native route.
func (a *Account) ExcelBPSModels() []string {
	if a == nil {
		return nil
	}
	raw, exists := a.Extra["openai_excel_bps_models"]
	if !exists {
		return []string{"gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra"}
	}
	var values []string
	switch list := raw.(type) {
	case []string:
		values = list
	case []any:
		for _, value := range list {
			if model, ok := value.(string); ok {
				values = append(values, model)
			}
		}
	}
	result := make([]string, 0, len(values))
	seen := map[string]bool{}
	for _, value := range values {
		model := strings.TrimSpace(value)
		if model != "" && !seen[model] {
			result = append(result, model)
			seen[model] = true
		}
	}
	return result
}

func (a *Account) UsesExcelBPSForModel(requestedModel string) bool {
	if !a.IsExcelBPSEnabled() {
		return false
	}
	model := strings.TrimSpace(a.GetMappedModel(requestedModel))
	for _, enabled := range a.ExcelBPSModels() {
		if model == enabled {
			return true
		}
	}
	return false
}

// Restore the account's saved native transport for an unselected model without
// mutating the shared scheduler snapshot or persisted flags. Selected models
// keep the existing BPS HTTP-only rules. Apply once, before model normalization.
func (a *Account) forOpenAIModel(requestedModel string) *Account {
	if !a.IsExcelBPSEnabled() || a.UsesExcelBPSForModel(requestedModel) {
		return a
	}
	scoped := *a
	scoped.Extra = make(map[string]any, len(a.Extra))
	for key, value := range a.Extra {
		scoped.Extra[key] = value
	}
	scoped.Extra["openai_excel_bps"] = false
	return &scoped
}

// A native websocket can change models on subsequent turns. Preserve the
// caller's mapping hook and reject a switch into an HTTP-only BPS model before
// sending that turn, including passthrough mode. Never mutate shared hooks.
func withExcelBPSModelGuard(account *Account, hooks *OpenAIWSIngressHooks) *OpenAIWSIngressHooks {
	if !account.IsExcelBPSEnabled() {
		return hooks
	}
	guarded := &OpenAIWSIngressHooks{}
	if hooks != nil {
		*guarded = *hooks
	}
	originalMap := guarded.MapRequestModel
	guarded.MapRequestModel = func(turn int, model string) (string, error) {
		mapped := model
		if originalMap != nil {
			value, err := originalMap(turn, model)
			if err != nil {
				return "", err
			}
			if strings.TrimSpace(value) != "" {
				mapped = value
			}
		}
		if account.UsesExcelBPSForModel(mapped) {
			return "", NewOpenAIWSClientCloseError(coderws.StatusPolicyViolation, "this model uses Excel BPS; send a new HTTP request to /v1/responses", nil)
		}
		return mapped, nil
	}
	return guarded
}
