package service

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestApplyCodexOAuthTransformCodeModeUsesDeveloperInputPrompt(t *testing.T) {
	reqBody := map[string]any{
		"model": "gpt-5.6-sol",
		"input": []any{map[string]any{"type": "message", "role": "user", "content": "hello"}},
	}

	result := applyCodexOAuthTransformWithOptions(reqBody, codexOAuthTransformOptions{UseCodeModeInstructions: true})
	require.NoError(t, result.Error)
	require.NotContains(t, reqBody, "instructions")
	input, ok := reqBody["input"].([]any)
	require.True(t, ok)
	require.Len(t, input, 2)
	first, ok := input[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "developer", first["role"])
	require.True(t, strings.HasPrefix(first["content"].(string), "You are Codex"))

	second := applyCodexOAuthTransformWithOptions(reqBody, codexOAuthTransformOptions{UseCodeModeInstructions: true})
	require.NoError(t, second.Error)
	input, ok = reqBody["input"].([]any)
	require.True(t, ok)
	require.Len(t, input, 2)
}
