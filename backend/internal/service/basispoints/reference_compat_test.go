package basispoints

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDisabledHostedToolsStillCheckActualNativeMedia(t *testing.T) {
	for _, tc := range []struct{ body, reason string }{
		{"{\"tool_choice\":\"none\",\"tools\":[{\"type\":\"web_search\"}]}", ""},
		{"{\"tool_choice\":\"none\",\"input\":[{\"type\":\"additional_tools\",\"tools\":[{\"type\":\"image_generation\"}]}]}", ""},
		{"{\"tool_choice\":\"none\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_file\",\"file_id\":\"file-a\"}]}]}", "native_media"},
		{"{\"tool_choice\":\"none\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_audio\"}]}]}", "native_media"},
	} {
		require.Equal(t, tc.reason, NativeFallbackReason([]byte(tc.body)))
	}
}
func TestCommandTransportDisplayPrefixAndRepairMetadata(t *testing.T) {
	source := testSource()
	source["tools"] = []any{functionCmdTestTool("exec_command")}
	_, b := mustPrepare(t, source, "scope", nil)
	call, err := b.translateCall(functionCmdTestNative(t, "functions.exec_command", "echo hi", "{}"))
	require.NoError(t, err)
	require.Equal(t, "exec_command", call["name"])
	original := repairCall("bad", "missing marker", "echo hi")
	args := transportArguments(original)
	args["extended_summary"] = "{\"workdir\":\"safe\"}"
	original["arguments"] = args
	same := functionCmdTestNative(t, "exec_command", "echo hi", "{\"workdir\":\"safe\"}")
	changed := functionCmdTestNative(t, "exec_command", "echo hi", "{\"workdir\":\"other\"}")
	require.True(t, b.preservesToolOperations([]object{original}, []object{same}))
	require.False(t, b.preservesToolOperations([]object{original}, []object{changed}))
}
func TestMixedRepairCannotChangeToolTargetOrMutateCorrection(t *testing.T) {
	_, b := repairBridge(t, nil)
	original := []object{repairCall("bad", "Run", "text(1)"), nativeCall(object{"name": "shell", "arguments": object{"cmd": "pwd"}})}
	correction := []object{repairCall("fixed", "codex2api.custom/functions.exec", "text(1)"), nativeCall(object{"name": "shell", "arguments": object{"cmd": "ls"}})}
	unchanged := fingerprint(correction)
	restored := b.restoreValidatedMixedOperations(original, correction)
	require.Equal(t, unchanged, fingerprint(correction))
	require.True(t, b.preservesToolOperations(original, restored))
	correction[1] = repairCall("wrong", "codex2api.custom/functions.exec", "text(999)")
	restored = b.restoreValidatedMixedOperations(original, correction)
	require.False(t, b.preservesToolOperations(original, restored))
}
