package basispoints

import (
	"io"
	"reflect"
	"strings"
	"testing"
)

// Model host-display prefixes must resolve identically for wrapped and direct
// calls. The exact declared name always takes precedence over a display alias.
func TestWrappedCatalogHostPrefix(t *testing.T) {
	for _, kind := range []string{"function", "custom"} {
		source := testSource()
		source["tools"] = []any{object{"type": kind, "name": "shell", "parameters": object{"type": "object"}}}
		cache := new(ReplayCache)
		_, bridge := mustPrepare(t, source, "prefix-test", cache)
		envelope := object{"name": "functions.shell"}
		if kind == "function" {
			envelope["arguments"] = object{"command": "echo test"}
		} else {
			envelope["input"] = "verbatim \"text\"\n"
		}
		native := nativeCall(envelope)
		call, err := bridge.translateCall(native)
		if err != nil {
			t.Fatalf("%s: host prefix incorrectly rejected: %v", kind, err)
		}
		if call["name"] != "shell" || call["namespace"] != nil {
			t.Fatalf("wrong identity: %#v", call)
		}
		if !reflect.DeepEqual(cache.get(bridge.scope, text(call["call_id"])), native) {
			t.Fatal("original wrapper must be retained for replay")
		}
		if kind == "custom" && call["input"] != envelope["input"] {
			t.Fatal("custom input changed")
		}
	}
}

func TestWrappedCatalogPrefixExactNameWins(t *testing.T) {
	source := testSource()
	source["tools"] = []any{
		object{"type": "function", "name": "shell"},
		object{"type": "namespace", "name": "functions", "tools": []any{object{"type": "function", "name": "shell"}}},
	}
	_, bridge := mustPrepare(t, source, "scope", nil)
	call, err := bridge.translateCall(nativeCall(object{"name": "functions.shell", "arguments": object{}}))
	if err != nil || call["namespace"] != "functions" {
		t.Fatalf("exact name lost: %#v %v", call, err)
	}
}

func TestCatalogHostPrefixNeverAdmitsUnknownTool(t *testing.T) {
	source := testSource()
	source["tools"] = []any{object{"type": "function", "name": "shell"}}
	_, bridge := mustPrepare(t, source, "scope", nil)
	for _, name := range []string{"functions.missing", "other.shell", "functions.functions.shell"} {
		native := nativeCall(object{"name": name, "arguments": object{"secret": "do-not-emit"}})
		stream := bridge.Stream(io.NopCloser(strings.NewReader(sse(object{"type": "response.completed", "response": object{"output": []any{native}}}))))
		raw, err := io.ReadAll(stream)
		_ = stream.Close()
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(raw), "response.failed") || strings.Contains(string(raw), "response.function_call_arguments") || strings.Contains(string(raw), "do-not-emit") {
			t.Fatalf("unknown tool was not rejected safely: %s", raw)
		}
	}
}

func TestRawCustomCatalogHostPrefix(t *testing.T) {
	source := testSource()
	source["tools"] = []any{object{"type": "custom", "name": "exec"}}
	_, bridge := mustPrepare(t, source, "scope", nil)
	native := nativeCall(object{})
	native["arguments"] = object{"summary": "codex2api.custom/functions.exec", "code": "exact code"}
	call, err := bridge.translateCall(native)
	if err != nil || call["name"] != "exec" || call["input"] != "exact code" {
		t.Fatalf("raw custom prefix: %#v %v", call, err)
	}
}
