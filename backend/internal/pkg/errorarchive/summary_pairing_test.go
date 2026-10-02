package errorarchive

import (
	"encoding/json"
	"github.com/tidwall/gjson"
	"strings"
	"testing"
)

func TestSummaryIncludesAdditionalToolsAndMissingHistoryWithoutContents(t *testing.T) {
	x := map[string]any{"input": []any{
		map[string]any{"type": "additional_tools", "tools": []any{map[string]any{"type": "custom", "name": "exec"}}},
		map[string]any{"type": "custom_tool_call", "call_id": "private-paired-id", "name": "exec", "input": "secret-script"},
		map[string]any{"type": "custom_tool_call_output", "call_id": "private-paired-id", "output": "secret-output"},
		map[string]any{"type": "custom_tool_call_output", "call_id": "private-missing-id", "output": "more-secret-output"},
	}}
	body, _ := json.Marshal(x)
	summary := RequestSummary(body)
	if gjson.GetBytes(summary, "tool_count").Int() != 1 || gjson.GetBytes(summary, "additional_tool_directories").Int() != 1 {
		t.Fatal("omitted additional tool catalog")
	}
	if gjson.GetBytes(summary, "tool_history_pairing.results_without_call_in_request").Int() != 1 || gjson.GetBytes(summary, "tool_history_pairing.samples.0.input_index").Int() != 3 {
		t.Fatal("wrong orphan location")
	}
	for _, secret := range []string{"private-paired-id", "private-missing-id", "secret-script", "secret-output"} {
		if strings.Contains(string(summary), secret) {
			t.Fatal("summary leaked private payload")
		}
	}
}
