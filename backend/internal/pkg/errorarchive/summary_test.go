package errorarchive

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestRequestSummaryCountsBeyondImagePrefixesWithoutPayload(t *testing.T) {
	var input []any
	for i := 0; i < 140; i++ {
		input = append(input, map[string]any{"role": "user", "content": []any{map[string]any{"type": "input_image", "image_url": "data:image/png;base64," + strings.Repeat("X", 40000)}, map[string]any{"type": "input_text", "text": "private-secret"}}})
	}
	input = append(input, map[string]any{"type": "function_call_output", "output": []any{map[string]any{"type": "encrypted_content", "encrypted_content": "private-blob"}}})
	body, _ := json.Marshal(map[string]any{"model": "test", "input": input, "tools": []any{map[string]any{"type": "namespace", "name": "functions", "tools": []any{map[string]any{"type": "custom", "name": "exec", "description": "private-instructions"}}}}})
	summary := RequestSummary(body)
	require.True(t, json.Valid(summary))
	require.Less(t, len(summary), 64<<10)
	require.Equal(t, int64(141), gjson.GetBytes(summary, "input_items").Int())
	require.Equal(t, int64(140), gjson.GetBytes(summary, "inline_images").Int())
	require.Equal(t, int64(1), gjson.GetBytes(summary, "content_types.encrypted_content").Int())
	require.Equal(t, "exec", gjson.GetBytes(summary, "tool_catalog.0.name").String())
	require.True(t, gjson.GetBytes(summary, "details_truncated").Bool())
	for _, secret := range []string{"private-secret", "private-blob", "private-instructions", "data:image", strings.Repeat("X", 100)} {
		require.NotContains(t, string(summary), secret)
	}
	ctx, trace := WithTrace(context.Background())
	defer trace.Release()
	for _, phase := range []string{"tool_validation", "tool_correction_rejected", "tool_correction_rejected"} {
		AddDiagnosticSummary(ctx, phase, summary, []byte(strings.Repeat("r", 400<<10)))
	}
	var diagnostics []Diagnostic
	require.NoError(t, json.Unmarshal(trace.Snapshot(), &diagnostics))
	require.Len(t, diagnostics, 3)
	for _, d := range diagnostics {
		require.False(t, d.Truncated)
		require.Empty(t, d.Request)
		require.JSONEq(t, string(summary), string(d.RequestSummary))
		require.Len(t, d.Response, 400<<10)
	}
}
