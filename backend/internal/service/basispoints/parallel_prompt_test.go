package basispoints

import (
	"strings"
	"testing"
)

func TestParallelPromptSeparatesIndependentReadsFromUnsafeCalls(t *testing.T) {
	source := testSource()
	source["tools"] = []any{
		object{"type": "function", "name": "read_file"},
		object{"type": "function", "name": "apply_patch"},
	}

	wire, _ := mustPrepare(t, source, "parallel-prompt", nil)
	items := mustTestValue[[]any](t, wire["input"])
	protocolMessage := mustTestValue[object](t, items[1])
	protocol := text(mustTestValue[[]any](t, protocolMessage["content"])[0].(object)["text"])
	for _, phrase := range []string{
		"independent and read-only",
		"same response",
		"do not wait for one result merely to schedule another",
		"state-changing tools",
		"Keep",
		"serial",
	} {
		if !strings.Contains(protocol, phrase) {
			t.Fatalf("parallel safety rule missing %q", phrase)
		}
	}

	source["parallel_tool_calls"] = false
	serialWire, _ := mustPrepare(t, source, "serial-prompt", nil)
	serialItems := mustTestValue[[]any](t, serialWire["input"])
	serialMessage := mustTestValue[object](t, serialItems[1])
	serialProtocol := text(mustTestValue[[]any](t, serialMessage["content"])[0].(object)["text"])
	if !strings.Contains(serialProtocol, "Call one client tool at a time.") {
		t.Fatal("serial client preference was not preserved")
	}
	if strings.Contains(serialProtocol, "same response as separate native run_officejs calls") {
		t.Fatal("parallel guidance leaked into a serial request")
	}
}
