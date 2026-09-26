package basispoints

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"
)

type matrixTool struct {
	Name string
	Kind string
}

func matrixTools(t *testing.T) []matrixTool {
	t.Helper()
	raw, err := os.ReadFile("testdata/exec_catalog_names.json")
	if err != nil {
		t.Fatal(err)
	}
	var tools []matrixTool
	if err = json.Unmarshal(raw, &tools); err != nil {
		t.Fatal(err)
	}
	if len(tools) < 64 {
		t.Fatal("catalog fixture unexpectedly shrank")
	}
	return tools
}

func matrixNames(name string) []string {
	names := []string{name, "functions." + name}
	if i := strings.LastIndex(name, "__"); strings.HasPrefix(name, "mcp__") && i > 3 {
		dotted := name[:i] + "." + name[i+2:]
		names = append(names, dotted, "functions."+dotted)
	}
	return names
}

func matrixNative(t *testing.T, name, mode string, value any) object {
	t.Helper()
	field := "arguments"
	if _, ok := value.(string); ok {
		field = "input"
	}
	n := nativeCall(object{"name": name, field: value})
	switch mode {
	case "direct":
		n["name"] = name
		if field == "input" {
			n["type"] = "custom_tool_call"
			n["input"] = value
			delete(n, "arguments")
		} else {
			raw, _ := json.Marshal(value)
			n["arguments"] = string(raw)
		}
	case "invocation":
		raw, _ := json.Marshal(value)
		n["arguments"] = object{"code": "await " + name + "(" + string(raw) + ");"}
	case "raw":
		if field == "input" {
			n["arguments"] = object{"summary": customTransportPrefix + name, "code": value}
		} else {
			args := value.(object)
			meta := object{}
			for k, v := range args {
				if k != "code" {
					meta[k] = v
				}
			}
			raw, _ := json.Marshal(meta)
			n = functionCodeTestNative(t, name, args["code"], string(raw))
		}
	}
	return n
}

func TestToolCatalogNameMatrix(t *testing.T) {
	count := 0
	for _, spec := range matrixTools(t) {
		for _, nested := range []bool{false, true} {
			source := testSource()
			var value any = object{"code": "literal 中文\n\"quotes\"\\path", "number": json.Number("9007199254740993")}
			kind := "function"
			if spec.Kind == "string" {
				kind = "custom"
				value = "*** Begin Patch\nliteral 中文 \\ \"quotes\"\n*** End Patch"
			}
			if nested {
				argType := "{ code: string; number: number }"
				if spec.Kind == "string" {
					argType = "string"
				}
				source = deferredExecSource("declare const tools: { " + spec.Name + "(args: " + argType + "): Promise<unknown>; };")
			} else {
				source["tools"] = []any{object{"type": kind, "name": spec.Name, "parameters": object{"type": "object", "properties": object{"code": object{"type": "string"}}}}}
			}
			for _, name := range matrixNames(spec.Name) {
				modes := []string{"wrapped", "direct", "invocation", "raw"}
				// Raw function-code transport needs a top-level explicit code schema.
				if nested && kind == "function" {
					modes = modes[:3]
				}
				for _, mode := range modes {
					count++
					t.Run(spec.Name+"/"+name+"/"+mode+map[bool]string{true: "/nested", false: "/catalog"}[nested], func(t *testing.T) {
						_, bridge := mustPrepare(t, source, "matrix", new(ReplayCache))
						native := matrixNative(t, name, mode, value)
						before, _ := json.Marshal(native)
						call, err := bridge.translateCall(native)
						if err != nil {
							t.Fatal(err)
						}
						after, _ := json.Marshal(native)
						if string(before) != string(after) {
							t.Fatal("native input changed")
						}
						if call["call_id"] != native["call_id"] {
							t.Fatal("call ID changed")
						}
						if nested {
							if call["name"] != "exec" || call["namespace"] != "functions" || call["type"] != "custom_tool_call" {
								t.Fatalf("bad relay %#v", call)
							}
							input := text(call["input"])
							prefix := "text(await tools[" + quoted(spec.Name) + "](JSON.parse("
							if !strings.HasPrefix(input, prefix) || !strings.HasSuffix(input, ")));") {
								t.Fatalf("bad nested input %s", input)
							}
							var encoded string
							if err := json.Unmarshal([]byte(strings.TrimSuffix(strings.TrimPrefix(input, prefix), ")));")), &encoded); err != nil {
								t.Fatal(err)
							}
							var decoded any
							if err := decode([]byte(encoded), &decoded); err != nil || !reflect.DeepEqual(decoded, value) {
								t.Fatalf("nested payload changed: %#v", decoded)
							}
						} else {
							if call["name"] != spec.Name {
								t.Fatalf("wrong name %v", call["name"])
							}
							if kind == "custom" {
								if call["input"] != value {
									t.Fatal("custom input changed")
								}
							} else {
								var got any
								_ = decode([]byte(text(call["arguments"])), &got)
								if !reflect.DeepEqual(got, value) {
									t.Fatal("function args changed")
								}
							}
						}
					})
				}
			}
		}
	}
	t.Logf("catalog tools=%d cases=%d", len(matrixTools(t)), count)
}

func TestCatalogMCPAliasesAreBidirectionalAndExactWins(t *testing.T) {
	for _, declared := range []string{"mcp__node_repl.js", "mcp__node_repl__js"} {
		source := testSource()
		source["tools"] = []any{functionCodeTestTool(declared)}
		_, bridge := mustPrepare(t, source, "", nil)
		for _, name := range matrixNames("mcp__node_repl__js") {
			call, err := bridge.translateCall(matrixNative(t, name, "wrapped", object{"code": "exact"}))
			if err != nil || call["name"] != declared {
				t.Fatalf("%s -> %s: %#v %v", name, declared, call, err)
			}
		}
	}
	source := testSource()
	source["tools"] = []any{functionCodeTestTool("mcp__node_repl.js"), functionCodeTestTool("mcp__node_repl__js")}
	_, bridge := mustPrepare(t, source, "", nil)
	for _, name := range []string{"mcp__node_repl.js", "mcp__node_repl__js"} {
		call, err := bridge.translateCall(matrixNative(t, name, "wrapped", object{"code": "exact"}))
		if err != nil || call["name"] != name {
			t.Fatal("exact match lost")
		}
	}
}

func TestToolCatalogNamespacedAndDeferredMatrix(t *testing.T) {
	count := 0
	for _, spec := range matrixTools(t) {
		if spec.Kind != "object" {
			continue
		}
		for _, catalogMode := range []string{"namespace", "additional_tools", "deferred"} {
			for _, name := range matrixNames(spec.Name) {
				for _, mode := range []string{"wrapped", "direct", "invocation"} {
					if catalogMode == "deferred" && !strings.HasPrefix(spec.Name, "mcp__") {
						continue
					}
					count++
					t.Run(catalogMode+"/"+name+"/"+mode, func(t *testing.T) {
						source := testSource()
						expectedNamespace, expectedName := "functions", spec.Name
						if catalogMode == "deferred" {
							source = deferredExecSource(deferredExecDescription)
							expectedName = "exec"
						} else {
							entry := object{"type": "namespace", "name": "functions", "tools": []any{functionCodeTestTool(spec.Name)}}
							if catalogMode == "additional_tools" {
								source["input"] = []any{object{"type": "additional_tools", "tools": []any{entry}}, message("user", "test")}
							} else {
								source["tools"] = []any{entry}
							}
						}
						_, bridge := mustPrepare(t, source, "namespaced-matrix", nil)
						call, err := bridge.translateCall(matrixNative(t, name, mode, object{"code": "literal"}))
						if err != nil {
							t.Fatal(err)
						}
						if call["name"] != expectedName || call["namespace"] != expectedNamespace {
							t.Fatalf("wrong target: %#v", call)
						}
					})
				}
			}
		}
	}
	t.Logf("namespaced/additional/deferred cases=%d", count)
}

func TestToolCatalogNameResolutionRejectsCollisionsAndUnsafeNames(t *testing.T) {
	source := testSource()
	source["tools"] = []any{
		functionCodeTestTool("mcp__alpha__beta.gamma"),
		functionCodeTestTool("mcp__alpha.beta__gamma"),
		object{"type": "namespace", "name": "one", "tools": []any{functionCodeTestTool("run")}},
		object{"type": "namespace", "name": "two", "tools": []any{functionCodeTestTool("run")}},
		functionCodeTestTool("shell"),
	}
	_, bridge := mustPrepare(t, source, "", nil)
	for _, name := range []string{"mcp__alpha__beta__gamma", "functions.mcp__alpha__beta__gamma", "run", "other.run", "tools.shell", "functions.functions.shell", "shell\n", "mcp__alpha.beta.gamma", "mcp__alpha.beta()"} {
		for _, mode := range []string{"wrapped", "direct", "invocation", "raw"} {
			// A newline before "(" is legal whitespace in the strict invocation
			// grammar; it is not part of its parsed tool identifier.
			if name == "shell\n" && mode == "invocation" {
				continue
			}
			if _, err := bridge.translateCall(matrixNative(t, name, mode, object{"code": "private"})); err == nil {
				t.Fatalf("accepted %q in %s", name, mode)
			}
		}
	}
}

func TestDirectCatalogCallPreservesSeparateNamespace(t *testing.T) {
	for _, namespace := range []string{"functions", "mcp__node_repl", "collaboration"} {
		source := testSource()
		source["tools"] = []any{object{"type": "namespace", "name": namespace, "tools": []any{functionCodeTestTool("inspect")}}, functionCodeTestTool("inspect")}
		_, bridge := mustPrepare(t, source, "", nil)
		native := matrixNative(t, "inspect", "direct", object{"code": "exact"})
		native["namespace"] = namespace
		call, err := bridge.translateCall(native)
		if err != nil || call["namespace"] != namespace {
			t.Fatalf("namespace %s lost: %#v %v", namespace, call, err)
		}
	}
	source := testSource()
	source["tools"] = []any{functionCodeTestTool("inspect")}
	_, bridge := mustPrepare(t, source, "", nil)
	for _, namespace := range []any{"other", " ", 42} {
		native := matrixNative(t, "inspect", "direct", object{"code": "exact"})
		native["namespace"] = namespace
		if _, err := bridge.translateCall(native); err == nil {
			t.Fatalf("unknown namespace ignored: %v", namespace)
		}
	}
}

func TestFullExecCatalogParallelStreamAndReplay(t *testing.T) {
	specs := matrixTools(t)
	var description strings.Builder
	for _, spec := range specs {
		argType := "{ code: string }"
		if spec.Kind == "string" {
			argType = "string"
		}
		fmt.Fprintf(&description, "declare const tools: { %s(args: %s): Promise<unknown>; };\n", spec.Name, argType)
	}
	source := deferredExecSource(description.String())
	source["input"] = []any{message("user", "Check catalog transport")}
	cache := new(ReplayCache)
	_, bridge := mustPrepare(t, source, "parallel-catalog", cache)
	var output []any
	for index, spec := range specs {
		var value any = object{"code": "literal"}
		if spec.Kind == "string" {
			value = "raw patch fixture"
		}
		names := matrixNames(spec.Name)
		native := matrixNative(t, names[len(names)-1], "wrapped", value)
		native["id"], native["call_id"] = fmt.Sprintf("fc_matrix_%d", index), fmt.Sprintf("call_matrix_%d", index)
		output = append(output, native)
	}
	body := bridge.Stream(io.NopCloser(strings.NewReader(sse(object{"type": "response.completed", "response": object{"output": output}}))))
	raw, err := io.ReadAll(body)
	_ = body.Close()
	if err != nil {
		t.Fatal(err)
	}
	var completed object
	dispatches := 0
	err = readEvents(strings.NewReader(string(raw)), func(_ string, data []byte) error {
		var event object
		if err := decode(data, &event); err != nil {
			return err
		}
		if event["type"] == "response.failed" {
			return fmt.Errorf("matrix stream failed")
		}
		if event["type"] == "response.custom_tool_call_input.done" {
			dispatches++
		}
		if event["type"] == "response.completed" {
			completed = event["response"].(object)
		}
		return nil
	})
	if err != nil || dispatches != len(specs) || completed == nil {
		t.Fatalf("dispatches=%d error=%v", dispatches, err)
	}
	input := source["input"].([]any)
	calls := completed["output"].([]any)
	input = append(input, calls...)
	for _, value := range calls {
		call := value.(object)
		input = append(input, object{"type": "custom_tool_call_output", "call_id": call["call_id"], "output": "fixture result"})
	}
	source["input"] = input
	first, _ := mustPrepare(t, source, "parallel-catalog", cache)
	again, _ := mustPrepare(t, source, "parallel-catalog", cache)
	if !reflect.DeepEqual(first, again) {
		t.Fatal("parallel replay unstable")
	}
	if first["metadata"].(object)["agent_iteration"] != "2" {
		t.Fatal("parallel results counted as many rounds")
	}
	_, _ = mustPrepare(t, source, "parallel-catalog", new(ReplayCache))
	source["parallel_tool_calls"] = false
	_, disabled := mustPrepare(t, source, "parallel-catalog", nil)
	// Recreate originals because completed conversion replaces output entries.
	two := []any{matrixNative(t, "exec_command", "wrapped", object{"code": "x"}), matrixNative(t, "get_goal", "wrapped", object{"code": "y"})}
	two[1].(object)["id"], two[1].(object)["call_id"] = "fc_other", "call_other"
	if err := disabled.translateResponse(object{"output": two}); err == nil {
		t.Fatal("parallel opt-out ignored")
	}
}

func TestDirectOnlyCatalogToolsStayOutsideExec(t *testing.T) {
	for _, name := range []string{"spawn_agent", "send_message", "followup_task", "interrupt_agent", "list_agents", "wait_agent"} {
		source := deferredExecSource(deferredExecDescription)
		source["tools"] = append(source["tools"].([]any), object{"type": "namespace", "name": "collaboration", "tools": []any{object{"type": "function", "name": name}}})
		for _, prefix := range []string{"", "functions."} {
			for _, mode := range []string{"wrapped", "direct", "invocation"} {
				_, bridge := mustPrepare(t, source, "direct-only", nil)
				call, err := bridge.translateCall(matrixNative(t, prefix+"collaboration."+name, mode, object{"message": "literal"}))
				if err != nil || call["type"] != "function_call" || call["namespace"] != "collaboration" || call["name"] != name {
					t.Fatalf("direct-only %s lost: %#v %v", name, call, err)
				}
			}
		}
	}
	for _, name := range []string{"mcp__cua_repl.js", "mcp__node_repl.js", "functions.wait", "functions.request_user_input"} {
		source := deferredExecSource(deferredExecDescription)
		source["tools"] = append(source["tools"].([]any), functionCodeTestTool(name))
		_, bridge := mustPrepare(t, source, "direct-only", nil)
		for _, mode := range []string{"wrapped", "direct", "invocation", "raw"} {
			call, err := bridge.translateCall(matrixNative(t, name, mode, object{"code": "literal", "description": "test"}))
			if err != nil || call["type"] != "function_call" || call["name"] != name {
				t.Fatalf("exact direct tool %s changed: %#v %v", name, call, err)
			}
		}
	}
}

func TestNestedStringContractRejectsObjectsAndRetainsReplay(t *testing.T) {
	source := deferredExecSource("declare const tools: { apply_patch(input: string): Promise<unknown>; };")
	cache := new(ReplayCache)
	_, bridge := mustPrepare(t, source, "patch-replay", cache)
	input := "*** Begin Patch\n*** Add File: fixture.txt\n+literal 中文\n*** End Patch"
	for _, mode := range []string{"wrapped", "direct", "invocation", "raw"} {
		native := matrixNative(t, "functions.apply_patch", mode, input)
		call, err := bridge.translateCall(native)
		if err != nil {
			t.Fatal(err)
		}
		source["input"] = []any{message("user", "apply"), call, object{"type": "custom_tool_call_output", "call_id": call["call_id"], "output": "ok"}}
		for _, replay := range []*ReplayCache{cache, new(ReplayCache)} {
			wire, _ := mustPrepare(t, source, "patch-replay", replay)
			items := wire["input"].([]any)
			result := items[len(items)-1].(object)
			if result["type"] != "function_call_output" || result["output"] != "ok" {
				t.Fatal("string tool replay result changed")
			}
		}
	}
	for _, bad := range []object{
		{"name": "apply_patch", "arguments": object{"input": input}},
		{"name": "apply_patch", "input": input, "args": object{}},
		{"name": "apply_patch", "input": object{"patch": input}},
	} {
		if _, err := bridge.translateCall(nativeCall(bad)); err == nil {
			t.Fatal("string contract accepted an object or mixed inputs")
		}
	}
}
