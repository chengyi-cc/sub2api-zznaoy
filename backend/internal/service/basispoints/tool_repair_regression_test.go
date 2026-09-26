package basispoints

import (
	"context"
	"io"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestToolRepairSerialReturnsOnlyUnchangedFirstOperation(t *testing.T) {
	for _, change := range []bool{false, true} {
		_, b := repairBridge(t, new(ReplayCache))
		b.parallelTools = false
		first := repairCall("first", "codex2api.custom/functions.exec", "text(1)")
		second := repairCall("second", "codex2api.custom/functions.exec", "text(2)")
		initial := repairResponse("resp_orig", 10, 2, first, second)
		calls := 0
		body := b.StreamWithToolRepair(context.Background(), io.NopCloser(strings.NewReader(sse(object{"type": "response.completed", "response": initial}))), func(ctx context.Context, failed object, validation error) (object, error) {
			calls++
			require.Contains(t, validation.Error(), "parallel")
			code := "text(1)"
			if change {
				code = "text(2)"
			}
			return repairResponse("resp_fix", 12, 3, repairCall("fixed", "codex2api.custom/functions.exec", code)), nil
		})
		events := repairEvents(t, body)
		require.Equal(t, 1, calls)
		last := events[len(events)-1]
		want := "response.completed"
		if change {
			want = "response.failed"
		}
		require.Equal(t, want, last["type"])
		if !change {
			response := last["response"].(object)
			require.Len(t, response["output"], 1)
			require.Nil(t, b.replay.get(b.scope, "second"))
		}
	}
}
func TestToolRepairUnavailablePlanFallsBackToVisibleText(t *testing.T) {
	_, b := repairBridge(t, nil)
	initial := repairResponse("resp_plan", 2, 1, nativePlanTestItem(object{"plan": []any{object{"step": "Read files", "status": "pending"}}}))
	body := b.StreamWithToolRepair(context.Background(), io.NopCloser(strings.NewReader(sse(object{"type": "response.completed", "response": initial}))), func(context.Context, object, error) (object, error) {
		return repairResponse("resp_fixed", 2, 1, object{"id": "msg_plan", "type": "message", "role": "assistant", "status": "completed", "content": []any{object{"type": "output_text", "text": "Plan: read the files.", "annotations": []any{}}}}), nil
	})
	events := repairEvents(t, body)
	require.Equal(t, "response.completed", events[len(events)-1]["type"])
	textEvents := 0
	for _, e := range events {
		if e["type"] == "response.output_text.delta" {
			textEvents++
		}
		if item, ok := e["item"].(object); ok {
			require.NotEqual(t, "function_call", item["type"])
		}
	}
	require.Equal(t, 1, textEvents)
}
func TestToolRepairRestoresRawPayloadWithoutMutatingCorrection(t *testing.T) {
	_, b := repairBridge(t, nil)
	for _, code := range []string{"text(1)", `{"payload":1}`} {
		original := []object{repairCall("bad", "Run", code)}
		corrected := []object{repairCall("fixed", "codex2api.custom/functions.exec", "text(999)")}
		out, err := b.restoreRawToolPayloads(original, corrected)
		require.NoError(t, err)
		require.Equal(t, code, transportArguments(out[0])["code"])
		require.Equal(t, "text(999)", transportArguments(corrected[0])["code"])
		require.True(t, b.preservesToolOperations(original, out))
	}
}
func TestToolRepairNeverChangesMalformedEnvelopeArguments(t *testing.T) {
	_, b := repairBridge(t, nil)
	original := []object{repairCall("bad", "codex2api.custom/shell", `{"name":"shell","arguments":{"cmd":"pwd"}}`)}
	changed := []object{repairCall("fixed", "Run", `{"name":"shell","arguments":{"cmd":"rm -rf /"}}`)}
	require.False(t, b.preservesToolOperations(original, changed))
}
func TestToolRepairFailureObserverCapturesBeforeMutation(t *testing.T) {
	_, b := repairBridge(t, nil)
	count := 0
	b.ObserveToolFailure(func(response object, err error) {
		require.Error(t, err)
		count++
		require.Equal(t, "bad", response["output"].([]any)[0].(object)["call_id"])
	})
	body := b.Stream(io.NopCloser(strings.NewReader(sse(object{"type": "response.completed", "response": repairResponse("resp", 1, 1, repairCall("bad", "Run", "text(1)"))}))))
	events := repairEvents(t, body)
	require.Equal(t, 1, count)
	require.Equal(t, "response.failed", events[len(events)-1]["type"])
}

func TestToolRepairCannotChangeFunctionCodeMetadata(t *testing.T) {
	_, b, err := Prepare([]byte(`{"model":"gpt-5.6-sol","input":"test","tools":[{"type":"function","name":"client.run","parameters":{"type":"object","properties":{"code":{"type":"string"},"cwd":{"type":"string"}},"required":["code","cwd"]}}]}`), "test", nil)
	require.NoError(t, err)
	original := repairCall("bad", "missing marker", "text(1)")
	args := transportArguments(original)
	args["extended_summary"] = `{"cwd":"safe"}`
	original["arguments"] = args
	same := functionCodeTestNative(t, "client.run", "text(1)", `{"cwd":"safe"}`)
	changed := functionCodeTestNative(t, "client.run", "text(1)", `{"cwd":"different"}`)
	require.True(t, b.preservesToolOperations([]object{original}, []object{same}))
	require.False(t, b.preservesToolOperations([]object{original}, []object{changed}))
}
