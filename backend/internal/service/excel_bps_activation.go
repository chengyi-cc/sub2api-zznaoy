package service

import "time"

// ExcelBPSActivationExtra applies defaults only on a disabled-to-enabled transition.
// Explicit values in the activation request win. Later edits keep saved choices.
func ExcelBPSActivationExtra(extra map[string]any, wasEnabled bool, reason string) map[string]any {
	if extra == nil {
		return nil
	}
	out := make(map[string]any, len(extra)+5)
	for k, v := range extra {
		out[k] = v
	}
	enabled, _ := out["openai_excel_bps"].(bool)
	if !enabled && wasEnabled {
		out["openai_excel_bps_last_transition"] = map[string]any{"reason": "manual_disabled", "at": time.Now().UTC().Format(time.RFC3339Nano)}
	}
	if !enabled || wasEnabled {
		return out
	}
	for _, key := range []string{"openai_excel_bps_cache_creation_as_input", "openai_excel_bps_auto_disable_on_403", "openai_rpm_overflow"} {
		if _, exists := out[key]; !exists {
			out[key] = true
		}
	}
	if _, exists := out["base_rpm"]; !exists {
		out["base_rpm"] = 15
	}
	out["openai_excel_bps_last_transition"] = map[string]any{"reason": reason, "at": time.Now().UTC().Format(time.RFC3339Nano)}
	return out
}
