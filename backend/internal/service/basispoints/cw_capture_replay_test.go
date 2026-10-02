package basispoints

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"github.com/tidwall/gjson"
	"io"
	"os"
	"path/filepath"
	"testing"
)

// Opt-in: source stays outside the repository. Only captured data is decoded;
// no client tool, shell command or network request is executed.
func TestCWCapturedExecEnvelopeReplaysWithoutModelCorrection(t *testing.T) {
	path := os.Getenv("BPS_CW_TOOL_CAPTURE")
	if path == "" {
		t.Skip("manual capture replay")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var capture struct {
		RequestWireBase64 string
		Diagnostics       []struct {
			Phase                  string
			UpstreamResponseBase64 string
		}
	}
	var root map[string]json.RawMessage
	if err = json.Unmarshal(data, &root); err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(root["request_wire_base64"], &capture.RequestWireBase64); err != nil {
		t.Fatal(err)
	}
	var diagnostics []map[string]json.RawMessage
	if err = json.Unmarshal(root["diagnostics"], &diagnostics); err != nil {
		t.Fatal(err)
	}
	decodeDiag := func(index int) object {
		t.Helper()
		var encoded string
		if err := json.Unmarshal(diagnostics[index]["upstream_response_base64"], &encoded); err != nil {
			t.Fatal(err)
		}
		raw, e := base64.StdEncoding.DecodeString(encoded)
		if e != nil {
			t.Fatal(e)
		}
		var value object
		if e = decode(raw, &value); e != nil {
			t.Fatal(e)
		}
		return value
	}
	original := decodeDiag(0)["response"].(object)
	corrected := decodeDiag(1)["response"].(object)
	native := original["output"].([]any)[0].(object)
	fixed := corrected["output"].([]any)[0].(object)
	args := transportArguments(native)
	var envelope object
	if err = decode([]byte(text(args["code"])), &envelope); err != nil {
		t.Fatal(err)
	}
	code := text(envelope["arguments"].(object)["code"])
	if code != text(transportArguments(fixed)["code"]) {
		t.Fatal("capture correction changed raw payload")
	}
	wire, e := base64.StdEncoding.DecodeString(capture.RequestWireBase64)
	if e != nil {
		t.Fatal(e)
	}
	// The first additional_tools item is complete although the later image is cut.
	first := gjson.GetBytes(wire, "input.0")
	var catalog object
	if decode([]byte(first.Raw), &catalog) != nil || catalog["type"] != "additional_tools" {
		t.Fatal("complete captured tool catalog unavailable")
	}
	source := testSource()
	source["input"] = []any{catalog, message("user", "offline tool relay validation")}
	_, b := mustPrepare(t, source, "private-capture", new(ReplayCache))
	repairs := 0
	body := b.StreamWithToolRepair(context.Background(), io.NopCloser(bytes.NewBufferString(sse(object{"type": "response.completed", "response": original}))), func(context.Context, object, error) (object, error) {
		repairs++
		t.Fatal("unexpected corrective model request")
		return nil, nil
	})
	out, e := io.ReadAll(body)
	_ = body.Close()
	if e != nil {
		t.Fatal(e)
	}
	var final object
	if e = readEvents(bytes.NewReader(out), func(_ string, raw []byte) error {
		var event object
		if err := decode(raw, &event); err != nil {
			return err
		}
		if event["type"] == "response.completed" {
			final = event["response"].(object)
		}
		return nil
	}); e != nil {
		t.Fatal(e)
	}
	if final == nil || repairs != 0 {
		t.Fatal("captured response did not complete")
	}
	call := final["output"].([]any)[0].(object)
	if call["type"] != "custom_tool_call" || call["input"] != code {
		t.Fatal("raw script not preserved")
	}
	if output := os.Getenv("BPS_CW_AUDIT_OUTPUT"); output != "" {
		sum := sha256.Sum256([]byte(code))
		report, _ := json.MarshalIndent(map[string]any{"fixture": filepath.Base(path), "raw_script_bytes": len([]byte(code)), "raw_script_characters": len([]rune(code)), "payload_sha256": hex.EncodeToString(sum[:]), "correction_payload_identical": true, "model_correction_requests": 0, "client_tools_executed": 0, "replay_scope": "complete captured catalog and terminal tool item only; full request was truncated"}, "", "  ")
		if e = os.WriteFile(output, report, 0600); e != nil {
			t.Fatal(e)
		}
	}
}
