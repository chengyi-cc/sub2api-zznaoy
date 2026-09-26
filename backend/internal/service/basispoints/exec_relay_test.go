package basispoints

import (
	"context"
	"encoding/json"
	"io"
	"reflect"
	"strings"
	"testing"
)

func TestLegacyExecCommandPreservesArgumentsWithoutModelRepair(t *testing.T) {
	for _, direct := range []bool{false, true} {
		for _, encoded := range []bool{false, true} {
			source := execTestSource()
			_, bridge := mustPrepare(t, source, "legacy-exec", new(ReplayCache))
			args := object{"cmd": "Write-Output '你好'\n# literal: \"\\ ); throw 1; //", "shell": "powershell", "workdir": "C:/example folder", "yield_time_ms": json.Number("30000"), "max_output_tokens": json.Number("12000"), "login": false}
			var value any = args
			if encoded {
				raw, _ := json.Marshal(args)
				value = string(raw)
			}
			native := nativeCall(object{"name": "functions.exec", "arguments": value})
			if direct {
				native["name"], native["arguments"] = "functions.exec", value
			}
			call, err := bridge.translateCall(native)
			if err != nil {
				t.Fatal(err)
			}
			if call["type"] != "custom_tool_call" || call["name"] != "exec" || call["namespace"] != "functions" || call["call_id"] != native["call_id"] {
				t.Fatal("tool identity changed")
			}
			code := text(call["input"])
			prefix, suffix := "text(await tools[\"exec_command\"](JSON.parse(", ")));"
			if !strings.HasPrefix(code, prefix) || !strings.HasSuffix(code, suffix) {
				t.Fatal("unexpected transport")
			}
			var literal string
			if err := json.Unmarshal([]byte(strings.TrimSuffix(strings.TrimPrefix(code, prefix), suffix)), &literal); err != nil {
				t.Fatal(err)
			}
			var restored object
			if err := decode([]byte(literal), &restored); err != nil || !reflect.DeepEqual(restored, args) {
				t.Fatal("command arguments changed")
			}
			response := object{"id": "resp_legacy", "status": "completed", "output": []any{native}}
			err = bridge.translateCompleted(context.Background(), response, func(context.Context, object, error) (object, error) {
				t.Fatal("unexpected model repair")
				return nil, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			source["input"] = []any{message("user", "test"), call, object{"type": "custom_tool_call_output", "call_id": call["call_id"], "output": "ok"}}
			first, _ := mustPrepare(t, source, "legacy-exec", bridge.replay)
			again, _ := mustPrepare(t, source, "legacy-exec", bridge.replay)
			if !reflect.DeepEqual(first, again) {
				t.Fatal("unstable follow-up request")
			}
		}
	}
}

func TestLegacyExecCommandRequiresExactDeclaredContract(t *testing.T) {
	_, bridge := mustPrepare(t, execTestSource(), "", nil)
	for _, envelope := range []object{
		{"name": "functions.exec", "arguments": object{"cmd": "pwd", "code": "text(99)"}},
		{"name": "functions.exec", "arguments": object{"cmd": "pwd"}, "input": "text(99)"},
		{"name": "functions.exec", "arguments": object{"cmd": "pwd"}, "args": object{}},
		{"name": "functions.exec", "arguments": object{"cmd": json.Number("123")}},
		{"name": "functions.exec", "arguments": object{"cmd": "  "}},
	} {
		if _, err := bridge.translateCall(nativeCall(envelope)); err == nil {
			t.Fatal("accepted conflicting or invalid command")
		}
	}
	for _, description := range []string{"Example tools.exec_command({cmd:'pwd'})", "declare const tools: { exec_command(input: string): Promise<unknown>; };"} {
		source := testSource()
		source["tools"] = []any{object{"type": "custom", "name": "exec", "description": description}}
		_, b := mustPrepare(t, source, "", nil)
		if _, err := b.translateCall(nativeCall(object{"name": "exec", "arguments": object{"cmd": "pwd"}})); err == nil {
			t.Fatal("undeclared command contract accepted")
		}
	}
	// A marked custom payload is always raw source, even if it resembles legacy arguments.
	marked := repairCall("raw", "codex2api.custom/functions.exec", `{"cmd":"pwd"}`)
	call, err := bridge.translateCall(marked)
	if err != nil || call["input"] != `{"cmd":"pwd"}` {
		t.Fatal("raw custom input changed")
	}
}

const execTestDescription = "Run JavaScript orchestration. tools holds callable tools; text(value) displays output.\n" +
	"declare const tools: { exec_command(args: { cmd: string }): Promise<unknown>; };\n" +
	"declare const tools: { write_stdin(args: { session_id: number }): Promise<unknown>; };\n"

func execTestSource() object {
	source := testSource()
	source["tools"] = []any{object{"type": "namespace", "name": "functions", "tools": []any{
		object{"type": "custom", "name": "exec", "description": execTestDescription},
	}}}
	return source
}

// Live BPS reproduction: only functions.exec is exposed, but the model places
// its documented nested exec_command in the inner JSON envelope.
func TestExecRelayRecoversDeclaredNestedFunction(t *testing.T) {
	for _, direct := range []bool{false, true} {
		for _, name := range []string{"exec_command", "functions.exec_command"} {
			source := execTestSource()
			cache := new(ReplayCache)
			_, bridge := mustPrepare(t, source, "exec-relay", cache)
			args := object{"cmd": "Write-Output BPS_TEST_OK", "literal": "\"\\\n); process.exit(); //", "__proto__": object{"keep": true}}
			native := nativeCall(object{"name": name, "arguments": args})
			if direct {
				encoded, _ := json.Marshal(args)
				native["name"], native["arguments"] = name, string(encoded)
			}
			call, err := bridge.translateCall(native)
			if err != nil {
				t.Fatalf("direct=%v name=%s: %v", direct, name, err)
			}
			if call["type"] != "custom_tool_call" || call["name"] != "exec" || call["namespace"] != "functions" || call["call_id"] != native["call_id"] {
				t.Fatalf("lost identity: %#v", call)
			}
			input := text(call["input"])
			prefix, suffix := "text(await tools[\"exec_command\"](JSON.parse(", ")));"
			if !strings.HasPrefix(input, prefix) || !strings.HasSuffix(input, suffix) {
				t.Fatalf("unexpected relay code: %q", input)
			}
			var literal string
			if err := json.Unmarshal([]byte(strings.TrimSuffix(strings.TrimPrefix(input, prefix), suffix)), &literal); err != nil {
				t.Fatal(err)
			}
			var restored object
			if err := decode([]byte(literal), &restored); err != nil || !reflect.DeepEqual(restored, args) {
				t.Fatalf("arguments changed: %#v %v", restored, err)
			}
			if !direct && !reflect.DeepEqual(cache.get(bridge.scope, text(call["call_id"])), native) {
				t.Fatal("original wrapper not retained")
			}
			source["input"] = []any{message("user", "test"), call, object{"type": "custom_tool_call_output", "call_id": call["call_id"], "output": "BPS_TEST_OK"}}
			first, _ := mustPrepare(t, source, "exec-relay", cache)
			again, _ := mustPrepare(t, source, "exec-relay", cache)
			if !reflect.DeepEqual(first, again) {
				t.Fatal("follow-up request is unstable")
			}
			items := first["input"].([]any)
			result := items[len(items)-1].(object)
			if result["type"] != "function_call_output" || result["call_id"] != call["call_id"] || result["output"] != "BPS_TEST_OK" {
				t.Fatalf("result lost: %#v", result)
			}
			_, _ = mustPrepare(t, source, "exec-relay", new(ReplayCache))
		}
	}
}

func TestExecRelayRejectsUndeclaredAmbiguousAndInvalidCalls(t *testing.T) {
	for _, name := range []string{"missing", "other.exec_command", "functions.functions.exec_command", "read_ranges", "constructor"} {
		_, bridge := mustPrepare(t, execTestSource(), "", nil)
		if _, err := bridge.translateCall(nativeCall(object{"name": name, "arguments": object{}})); err == nil {
			t.Fatalf("accepted %q", name)
		}
	}
	for _, args := range []any{nil, []any{}, "null", "[]", "{} {}", "not JSON"} {
		_, bridge := mustPrepare(t, execTestSource(), "", nil)
		if _, err := bridge.translateCall(nativeCall(object{"name": "exec_command", "arguments": args})); err == nil {
			t.Fatalf("accepted arguments %#v", args)
		}
	}
	for _, description := range []string{"Example: tools.exec_command({cmd: 'pwd'});", "exec_command is available", "declare const other: { exec_command(args: object): unknown; };"} {
		source := testSource()
		source["tools"] = []any{object{"type": "custom", "name": "exec", "description": description}}
		_, bridge := mustPrepare(t, source, "", nil)
		if _, err := bridge.translateCall(nativeCall(object{"name": "exec_command", "arguments": object{}})); err == nil {
			t.Fatalf("prose became a declaration: %q", description)
		}
	}
	source := execTestSource()
	source["tools"] = append(source["tools"].([]any), object{"type": "custom", "name": "exec", "description": execTestDescription})
	_, bridge := mustPrepare(t, source, "", nil)
	if _, err := bridge.translateCall(nativeCall(object{"name": "exec_command", "arguments": object{}})); err == nil {
		t.Fatal("ambiguous exec hosts accepted")
	}
	source = execTestSource()
	source["tool_choice"] = "none"
	_, bridge = mustPrepare(t, source, "", nil)
	if _, err := bridge.translateCall(nativeCall(object{"name": "exec_command", "arguments": object{}})); err == nil {
		t.Fatal("tool_choice none bypassed")
	}
}

func TestExecRelayExactCatalogNameWins(t *testing.T) {
	source := execTestSource()
	source["tools"] = append(source["tools"].([]any), object{"type": "function", "name": "exec_command"})
	_, bridge := mustPrepare(t, source, "", nil)
	call, err := bridge.translateCall(nativeCall(object{"name": "exec_command", "arguments": object{}}))
	if err != nil || call["type"] != "function_call" || call["name"] != "exec_command" {
		t.Fatalf("exact declaration changed: %#v %v", call, err)
	}
}

func TestExecRelayCatalogAndHistoryStayDeterministic(t *testing.T) {
	for _, namespace := range []bool{false, true} {
		source := execTestSource()
		if !namespace {
			source["tools"] = []any{object{"type": "custom", "name": "exec", "description": execTestDescription}}
		}
		first, bridge := mustPrepare(t, source, "stable", nil)
		again, _ := mustPrepare(t, source, "stable", nil)
		if !reflect.DeepEqual(first, again) {
			t.Fatal("identical requests changed")
		}
		items := first["input"].([]any)
		protocol := items[1].(object)["content"].([]any)[0].(object)["text"].(string)
		if !strings.Contains(protocol, "nested JavaScript APIs, not separate catalog tools") {
			t.Fatal("missing exec guidance")
		}
		if strings.Contains(protocol, "Client tool \"exec_command\"") {
			t.Fatal("nested tool promoted into catalog")
		}
		call, err := bridge.translateCall(nativeCall(object{"name": "write_stdin", "arguments": object{"session_id": json.Number("123")}}))
		if err != nil || call["name"] != "exec" {
			t.Fatalf("second declared nested tool lost: %#v %v", call, err)
		}
		source["input"] = []any{message("user", "hello"), call, object{"type": "custom_tool_call_output", "call_id": call["call_id"], "output": "done"}}
		follow, _ := mustPrepare(t, source, "stable", nil)
		if !reflect.DeepEqual(items[:2], follow["input"].([]any)[:2]) {
			t.Fatal("stable prompt prefix moved on follow-up")
		}
		// Old replayable calls never grant a nested tool missing from today's catalog.
		source["tools"] = []any{object{"type": "custom", "name": "exec", "description": "No nested declarations."}}
		_, removed := mustPrepare(t, source, "stable", nil)
		if _, err := removed.translateCall(nativeCall(object{"name": "write_stdin", "arguments": object{}})); err == nil {
			t.Fatal("history granted a removed tool")
		}
	}
}

func TestExecRelayNeverInterpretsCustomInputOrConflictingArguments(t *testing.T) {
	_, bridge := mustPrepare(t, execTestSource(), "", nil)
	marked := nativeCall(object{})
	marked["arguments"] = object{"summary": customTransportPrefix + "exec_command", "code": "must-not-evaluate"}
	directCustom := object{"type": "custom_tool_call", "name": "exec_command", "id": "ctc_one", "call_id": "one", "input": "must-not-evaluate"}
	conflict := nativeCall(object{"name": "exec_command", "arguments": object{}, "args": object{}})
	mixed := nativeCall(object{"name": "exec_command", "arguments": object{}, "input": "must-not-evaluate"})
	oversized := nativeCall(object{"name": "exec_command", "arguments": object{"cmd": strings.Repeat("a", maxEnvelopeBytes)}})
	for _, native := range []object{marked, directCustom, conflict, mixed, oversized} {
		if _, err := bridge.translateCall(native); err == nil {
			t.Fatal("invalid nested transport accepted")
		}
	}
}

func TestExecRelayStreamsValidatedCallAndRejectsMixedTerminal(t *testing.T) {
	for _, mixed := range []bool{false, true} {
		_, bridge := mustPrepare(t, execTestSource(), "", nil)
		valid := nativeCall(object{"name": "exec_command", "arguments": object{"cmd": "echo ok"}})
		output := []any{valid}
		if mixed {
			bad := nativeCall(object{"name": "unknown", "arguments": object{"secret": "must-not-leak"}})
			bad["id"], bad["call_id"] = "fc_bad", "call_bad"
			output = append(output, bad)
		}
		body := bridge.Stream(io.NopCloser(strings.NewReader(sse(object{"type": "response.completed", "response": object{"output": output}}))))
		raw, err := io.ReadAll(body)
		_ = body.Close()
		if err != nil {
			t.Fatal(err)
		}
		if mixed {
			if !strings.Contains(string(raw), "response.failed") || strings.Contains(string(raw), "response.custom_tool_call_input") || strings.Contains(string(raw), "must-not-leak") {
				t.Fatalf("partial dispatch: %s", raw)
			}
		} else if strings.Contains(string(raw), "response.failed") || !strings.Contains(string(raw), "response.custom_tool_call_input.done") {
			t.Fatalf("relay not emitted: %s", raw)
		}
	}
}
