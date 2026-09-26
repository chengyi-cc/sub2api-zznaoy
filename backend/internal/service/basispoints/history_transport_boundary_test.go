package basispoints

import (
	"reflect"
	"testing"
)

func TestRebuiltCustomHistoryUsesCurrentRawContract(t *testing.T) {
	for _, namespace := range []string{"", "functions", "nested.tools"} {
		t.Run(namespace, func(t *testing.T) {
			for _, input := range []string{"", "const x = '中文';\r\n\ttext(x);\n", "{\"name\":\"not_a_target\",\"input\":\"opaque\"}"} {
				source := testSource()
				decl := object{"type": "custom", "name": "exec"}
				call := object{"type": "custom_tool_call", "name": "exec", "call_id": "call_original", "id": "ctc_original", "input": input}
				name := "exec"
				if namespace != "" {
					decl = object{"type": "namespace", "name": namespace, "tools": []any{decl}}
					call["namespace"] = namespace
					name = namespace + ".exec"
				}
				source["tools"] = []any{decl}
				source["input"] = []any{call, object{"type": "custom_tool_call_output", "call_id": "call_original", "output": "recorded result"}}
				var previous object
				cache := new(ReplayCache)
				for _, replay := range []*ReplayCache{nil, cache, cache, new(ReplayCache)} {
					body, bridge := mustPrepare(t, source, "history-boundary", replay)
					items := mustTestValue[[]any](t, body["input"])
					native := mustTestValue[object](t, items[len(items)-2])
					var outer object
					if err := decode([]byte(text(native["arguments"])), &outer); err != nil {
						t.Fatal(err)
					}
					if outer["summary"] != customTransportPrefix+name || outer["code"] != input {
						t.Fatal("rebuilt history teaches a transport different from the current raw custom contract")
					}
					restored, err := bridge.translateCall(native)
					if err != nil || historyCallFingerprint(restored) != historyCallFingerprint(call) {
						t.Fatalf("raw history changed the recorded client operation: %v", err)
					}
					if previous != nil && !reflect.DeepEqual(previous, body) {
						t.Fatal("history changed across replay-cache loss")
					}
					previous = body
				}
			}
		})
	}
}

func TestOriginalImageDetailIsPreserved(t *testing.T) {
	image := object{"type": "input_image", "image_url": "https://images.example/original.png", "detail": "original"}
	item := object{"type": "message", "role": "user", "content": []any{object{"type": "input_text", "text": "Inspect exact pixels."}, image}}
	source := testSource()
	source["input"] = []any{item}
	body, _ := mustPrepare(t, source, "image-boundary", nil)
	items := mustTestValue[[]any](t, body["input"])
	if !reflect.DeepEqual(items[len(items)-1], item) {
		t.Fatal("original image detail was downgraded or neighboring text changed")
	}
}

func TestRebuiltCustomHistoryKeepsCachedNativeVerbatim(t *testing.T) {
	source := testSource()
	source["tools"] = []any{object{"type": "custom", "name": "exec"}}
	cache := new(ReplayCache)
	_, bridge := mustPrepare(t, source, "cached-history", cache)
	native := nativeCall(object{"name": "exec", "input": "text('recorded only');"})
	native["provider_extension"] = object{"preserve": true}
	call, err := bridge.translateCall(native)
	if err != nil {
		t.Fatal(err)
	}
	source["input"] = []any{call, object{"type": "custom_tool_call_output", "call_id": call["call_id"], "output": "recorded"}}
	body, _ := mustPrepare(t, source, "cached-history", cache)
	items := mustTestValue[[]any](t, body["input"])
	if !reflect.DeepEqual(items[len(items)-2], native) {
		t.Fatal("existing native replay was rewritten during raw-contract canonicalization")
	}
}
