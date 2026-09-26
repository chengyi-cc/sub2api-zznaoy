package basispoints

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"testing"
)

func TestParallelToolStreamAndResultRoundtrip(t *testing.T) {
	for _, setting := range []any{nil, true, false} {
		t.Run(fmt.Sprint(setting), func(t *testing.T) {
			source := testSource()
			if setting != nil {
				source["parallel_tool_calls"] = setting
			}
			source["tools"] = []any{object{"type": "function", "name": "lookup"}, object{"type": "namespace", "name": "files", "tools": []any{object{"type": "custom", "name": "inspect"}}}}
			cache := new(ReplayCache)
			first, bridge := mustPrepare(t, source, "parallel-scope", cache)
			a := nativeCall(object{"name": "lookup", "arguments": object{"value": json.Number("9007199254740993")}})
			b := nativeCall(object{"name": "files.inspect", "input": "C:\\fixture\\test\nexact"})
			a["id"], a["call_id"] = "fc_first", "call_first"
			b["id"], b["call_id"] = "fc_second", "call_second"
			wire := sse(object{"type": "response.output_item.done", "item": a}) + sse(object{"type": "response.output_item.done", "item": b}) + sse(object{"type": "response.completed", "response": object{"status": "completed", "output": []any{object{"type": "reasoning", "id": "rs_1"}, a, b}}})
			stream := bridge.Stream(io.NopCloser(strings.NewReader(wire)))
			raw, err := io.ReadAll(stream)
			_ = stream.Close()
			if err != nil {
				t.Fatal(err)
			}
			if setting == false {
				if !strings.Contains(string(raw), "response.failed") || strings.Contains(string(raw), "response.function_call_arguments") || strings.Contains(string(raw), "response.custom_tool_call_input") {
					t.Fatalf("disabled parallel escaped: %s", raw)
				}
				return
			}
			var calls []any
			indices := []int{}
			terminal := 0
			lastSequence := -1
			err = readEvents(strings.NewReader(string(raw)), func(_ string, data []byte) error {
				var event object
				if err := decode(data, &event); err != nil {
					return err
				}
				n, _ := event["sequence_number"].(json.Number).Int64()
				if int(n) <= lastSequence {
					t.Fatal("non-monotonic sequence")
				}
				lastSequence = int(n)
				if event["type"] == "response.output_item.done" {
					calls = append(calls, event["item"])
					i, _ := event["output_index"].(json.Number).Int64()
					indices = append(indices, int(i))
				}
				if event["type"] == "response.completed" {
					terminal++
					response := event["response"].(object)
					if response["parallel_tool_calls"] != true {
						t.Fatal("parallel flag lost")
					}
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			if len(calls) != 2 || terminal != 1 || indices[0] != 1 || indices[1] != 2 {
				t.Fatalf("lost calls or positions: %v, terminal=%d", indices, terminal)
			}
			firstCall := calls[0].(object)
			secondCall := calls[1].(object)
			if firstCall["call_id"] != "call_first" || secondCall["call_id"] != "call_second" || secondCall["namespace"] != "files" || secondCall["input"] != "C:\\fixture\\test\nexact" {
				t.Fatal("call identity or input changed")
			}
			source["input"] = []any{message("user", "hello"), calls[0], calls[1], object{"type": "function_call_output", "call_id": "call_first", "output": "one"}, object{"type": "custom_tool_call_output", "call_id": "call_second", "output": "two"}}
			replay, _ := mustPrepare(t, source, "parallel-scope", cache)
			metadata := replay["metadata"].(object)
			if metadata["agent_iteration"] != "2" || metadata["task_id"] != first["metadata"].(object)["task_id"] {
				t.Fatalf("parallel results counted as extra rounds: %#v", metadata)
			}
			items := replay["input"].([]any)
			for i, native := range []object{a, b} {
				item := items[len(items)-4+i].(object)
				if item["call_id"] != native["call_id"] || item["arguments"] != native["arguments"] {
					t.Fatal("original upstream call lost in replay")
				}
			}
		})
	}
}

func TestParallelToolValidationIsAtomic(t *testing.T) {
	for _, failure := range []string{"unknown", "duplicate_call", "duplicate_item"} {
		source := testSource()
		source["tools"] = []any{object{"type": "function", "name": "lookup"}}
		_, bridge := mustPrepare(t, source, "scope", nil)
		a := nativeCall(object{"name": "lookup", "arguments": object{}})
		b := nativeCall(object{"name": "lookup", "arguments": object{}})
		b["id"], b["call_id"] = "fc_b", "call_b"
		switch failure {
		case "unknown":
			b = nativeCall(object{"name": "missing", "arguments": object{}})
			b["id"], b["call_id"] = "fc_b", "call_b"
		case "duplicate_call":
			b["call_id"] = a["call_id"]
		case "duplicate_item":
			b["id"] = a["id"]
		}
		stream := bridge.Stream(io.NopCloser(strings.NewReader(sse(object{"type": "response.completed", "response": object{"output": []any{a, b}}}))))
		raw, err := io.ReadAll(stream)
		_ = stream.Close()
		if err != nil || !strings.Contains(string(raw), "response.failed") || strings.Contains(string(raw), "response.function_call_arguments") {
			t.Fatalf("%s: partially dispatched %s, %v", failure, raw, err)
		}
	}
}

func TestParallelToolFlagRejectsInvalidType(t *testing.T) {
	source := testSource()
	source["parallel_tool_calls"] = "false"
	raw, _ := json.Marshal(source)
	if _, _, err := Prepare(raw, "scope", nil); err == nil {
		t.Fatal("invalid parallel flag accepted")
	}
}
