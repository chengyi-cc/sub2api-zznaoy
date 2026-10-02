package basispoints

import (
	"reflect"
	"testing"
)

func TestLegacyExecCodeEnvelopePreservesRawScript(t *testing.T) {
	source := deferredExecSource(deferredExecDescription)
	_, b := mustPrepare(t, source, "legacy-code", new(ReplayCache))
	code := "const p = \"D:\\\\fixture\\\\app\";\ntext(\"中文 \\\"quotes\\\"\");"
	envelope := object{"name": "functions.exec", "arguments": object{"code": code}}
	native := nativeCall(envelope)
	before := copyToolObject(native)
	call, err := b.translateCall(native)
	if err != nil {
		t.Fatal(err)
	}
	if call["type"] != "custom_tool_call" || call["input"] != code {
		t.Fatal("legacy wrapper changed raw code")
	}
	if !reflect.DeepEqual(before, native) {
		t.Fatal("native replay mutated")
	}
	corrected := copyToolObject(native)
	corrected["arguments"] = object{"summary": "codex2api.custom/functions.exec", "code": code}
	if !b.preservesToolOperations([]object{before}, []object{corrected}) {
		t.Fatal("identical operation incorrectly rejected")
	}
	corrected["arguments"] = object{"summary": "codex2api.custom/functions.exec", "code": code + "text('different');"}
	if b.preservesToolOperations([]object{before}, []object{corrected}) {
		t.Fatal("changed operation accepted")
	}
}

func TestLegacyExecCodeEnvelopeRejectsAmbiguousPayloads(t *testing.T) {
	source := deferredExecSource(deferredExecDescription)
	_, b := mustPrepare(t, source, "legacy-invalid", nil)
	for _, envelope := range []object{
		{"name": "functions.exec", "arguments": object{"code": "text(1)", "timeout": 1}},
		{"name": "functions.exec", "arguments": object{"code": 123}},
		{"name": "functions.exec", "arguments": object{"code": "text(1)"}, "input": "text(2)"},
		{"name": "functions.exec", "arguments": object{"code": "text(1)"}, "input": nil},
	} {
		if _, err := b.translateCall(nativeCall(envelope)); err == nil {
			t.Fatal("ambiguous code envelope accepted")
		}
	}
	plain := testSource()
	plain["tools"] = []any{object{"type": "custom", "name": "exec"}}
	_, unknown := mustPrepare(t, plain, "not-js-exec", nil)
	if _, err := unknown.translateCall(nativeCall(object{"name": "exec", "arguments": object{"code": "opaque"}})); err == nil {
		t.Fatal("arbitrary custom tool reinterpreted")
	}
}
