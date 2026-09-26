package basispoints

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

const deferredExecDescription = execTestDescription + "\nNested tools use normalized JavaScript identifiers and are listed in ALL_TOOLS."

func deferredExecSource(description string) object {
	source := execTestSource()
	source["tools"].([]any)[0].(object)["tools"].([]any)[0].(object)["description"] = description
	return source
}

// A second-step live reproduction returned mcp__node_repl.js, while Codex's
// exec runtime exposes mcp__node_repl__js. Deferred tools may not have a type
// declaration in the initial exec description at all.
func TestExecRelayMCPNamesAndDeferredRuntimeCatalog(t *testing.T) {
	for _, deferred := range []bool{false, true} {
		description := execTestDescription + "\ndeclare const tools: { mcp__node_repl__js(args: { code: string }): Promise<unknown>; };"
		if deferred {
			description = deferredExecDescription
		}
		for _, name := range []string{"mcp__node_repl.js", "functions.mcp__node_repl.js", "mcp__node_repl__js"} {
			_, bridge := mustPrepare(t, deferredExecSource(description), "scope", nil)
			call, err := bridge.translateCall(nativeCall(object{"name": name, "arguments": object{"code": "nodeRepl.write('ok')"}}))
			if err != nil {
				t.Fatalf("deferred=%v name=%s: %v", deferred, name, err)
			}
			if call["type"] != "custom_tool_call" || call["namespace"] != "functions" || call["name"] != "exec" {
				t.Fatalf("wrong client tool: %#v", call)
			}
			input := text(call["input"])
			if !strings.Contains(input, "mcp__node_repl__js") {
				t.Fatalf("lost normalized method: %s", input)
			}
			if deferred && (!strings.Contains(input, "ALL_TOOLS.filter") || !strings.Contains(input, "tool_not_available")) {
				t.Fatal("deferred call skipped runtime catalog validation")
			}
		}
	}
}

func TestExecRelayDeferredRequiresExplicitRuntimeContract(t *testing.T) {
	for _, description := range []string{execTestDescription, "An exec tool", "ALL_TOOLS may be available"} {
		_, bridge := mustPrepare(t, deferredExecSource(description), "", nil)
		if _, err := bridge.translateCall(nativeCall(object{"name": "mcp__node_repl.js", "arguments": object{}})); err == nil {
			t.Fatal("undeclared runtime contract accepted")
		}
	}
	_, bridge := mustPrepare(t, deferredExecSource(deferredExecDescription), "", nil)
	for _, name := range []string{"missing", "other.exec_command", "collaboration.spawn_agent", "mcp__node_repl.js.extra", "mcp__node_repl.js()", "mcp__node_repl.constructor", "functions.functions.mcp__node_repl.js"} {
		if _, err := bridge.translateCall(nativeCall(object{"name": name, "arguments": object{}})); err == nil {
			t.Fatalf("invalid deferred name accepted: %q", name)
		}
	}
}

func TestExecRelayDeferredPreservesEscapedArgumentsAndExactTools(t *testing.T) {
	source := deferredExecSource(deferredExecDescription)
	_, bridge := mustPrepare(t, source, "", nil)
	args := object{"code": "nodeRepl.write(\"中文\\n\"); // ); injected();", "timeout_ms": json.Number("30000")}
	first, err := bridge.translateCall(nativeCall(object{"name": "mcp__node_repl.js", "arguments": args}))
	if err != nil {
		t.Fatal(err)
	}
	second, err := bridge.translateCall(nativeCall(object{"name": "mcp__node_repl.js", "arguments": args}))
	if err != nil || first["input"] != second["input"] {
		t.Fatal("deferred conversion is not stable")
	}
	source["tools"] = append(source["tools"].([]any), object{"type": "function", "name": "mcp__node_repl.js"})
	_, bridge = mustPrepare(t, source, "", nil)
	call, err := bridge.translateCall(nativeCall(object{"name": "mcp__node_repl.js", "arguments": args}))
	if err != nil || call["type"] != "function_call" || call["name"] != "mcp__node_repl.js" {
		t.Fatal("deferred recovery replaced an exact declaration")
	}
}

func TestExecRelayShellThenDeferredMCPRoundtrip(t *testing.T) {
	source := deferredExecSource(deferredExecDescription)
	source["input"] = []any{message("user", "Run a shell check, then check its result in Node.")}
	cache := new(ReplayCache)
	for index, step := range []struct {
		name   string
		args   object
		result string
	}{
		{"exec_command", object{"cmd": "Write-Output OK"}, "OK"},
		{"mcp__node_repl.js", object{"code": "nodeRepl.write({ok:true})"}, "{\"ok\":true}"},
	} {
		_, bridge := mustPrepare(t, source, "multi-step", cache)
		native := nativeCall(object{"name": step.name, "arguments": step.args})
		native["call_id"], native["id"] = "call_"+step.name, "fc_"+step.name
		call, err := bridge.translateCall(native)
		if err != nil {
			t.Fatalf("step %d: %v", index, err)
		}
		input := source["input"].([]any)
		source["input"] = append(input, call, object{"type": "custom_tool_call_output", "call_id": call["call_id"], "output": step.result})
		follow, _ := mustPrepare(t, source, "multi-step", cache)
		again, _ := mustPrepare(t, source, "multi-step", cache)
		if !reflect.DeepEqual(follow, again) {
			t.Fatalf("step %d changed on replay", index)
		}
		items := follow["input"].([]any)
		if !reflect.DeepEqual(items[len(items)-2], native) {
			t.Fatalf("step %d lost native identity", index)
		}
		result := items[len(items)-1].(object)
		if result["type"] != "function_call_output" || result["call_id"] != call["call_id"] || result["output"] != step.result {
			t.Fatalf("step %d lost result", index)
		}
		_, _ = mustPrepare(t, source, "multi-step", new(ReplayCache))
	}
}
