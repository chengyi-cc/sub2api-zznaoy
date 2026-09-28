package basispoints

import (
	"bytes"
	"context"
	"github.com/stretchr/testify/require"
	"strings"
	"testing"
)

func TestLifecycleSeparatesToolReadyFromEmissionAcrossCompaction(t *testing.T) {
	_, bridge := repairBridge(t, nil)
	tool := nativeCall(object{"name": "shell", "arguments": object{"cmd": "pwd"}})
	compact := object{"type": "compaction", "id": "cmp_1", "encrypted_content": "private"}
	msg := object{"type": "message", "id": "msg_1", "role": "assistant", "content": []any{object{"type": "output_text", "text": "ready"}}}
	wire := sse(object{"type": "response.output_item.done", "item": tool}) + sse(object{"type": "response.output_item.done", "item": msg}) + sse(object{"type": "response.output_item.added", "item": compact}) + sse(object{"type": "response.output_item.done", "item": compact}) + sse(object{"type": "response.completed", "response": repairResponse("resp_1", 10, 2, tool, msg, compact)})
	var out bytes.Buffer
	var stages []StreamStage
	bridge.ObserveLifecycle(func(stage StreamStage) {
		if stage == StreamCompactionStarted {
			require.NotContains(t, stages, StreamToolEmitted)
			require.NotContains(t, out.String(), "response.function_call_arguments")
		}
		stages = append(stages, stage)
	})
	require.NoError(t, bridge.transformWithRepair(context.Background(), strings.NewReader(wire), &out, nil))
	require.Equal(t, []StreamStage{StreamFirstEvent, StreamToolReady, StreamMessageCompleted, StreamFirstOutput, StreamCompactionStarted, StreamCompactionCompleted, StreamUpstreamCompleted, StreamValidationStarted, StreamValidationCompleted, StreamToolEmitted}, stages)
	require.Contains(t, out.String(), "response.function_call_arguments.done")
}

func TestLifecycleReportsRepairWithoutDispatchingInvalidTool(t *testing.T) {
	_, bridge := repairBridge(t, nil)
	invalid := repairCall("bad", "Run client tool", "text(1);")
	wire := sse(object{"type": "response.completed", "response": repairResponse("resp_bad", 10, 2, invalid)})
	var stages []StreamStage
	bridge.ObserveLifecycle(func(stage StreamStage) { stages = append(stages, stage) })
	var out bytes.Buffer
	err := bridge.transformWithRepair(context.Background(), strings.NewReader(wire), &out, func(context.Context, object, error) (object, error) {
		require.Equal(t, StreamRepairStarted, stages[len(stages)-1])
		require.NotContains(t, stages, StreamToolEmitted)
		return repairResponse("resp_fixed", 10, 2, repairCall("fixed", "codex2api.custom/functions.exec", "text(1);")), nil
	})
	require.NoError(t, err)
	require.Equal(t, []StreamStage{StreamFirstEvent, StreamUpstreamCompleted, StreamValidationStarted, StreamRepairStarted, StreamRepairCompleted, StreamValidationCompleted, StreamToolEmitted}, stages)
}
