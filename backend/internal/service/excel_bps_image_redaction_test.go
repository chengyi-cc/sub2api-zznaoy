package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

func imageRedactorFixture(t *testing.T) (*excelImageRedactor, string, string) {
	t.Helper()
	token := strings.Repeat("0123456789abcdef", 4)
	link := "https://images.example" + ExcelBPSImagePath + "?token=" + token
	body, _ := json.Marshal(map[string]string{"image_url": link})
	r := newExcelImageRedactor(body)
	require.NotNil(t, r)
	return r, link, token
}

func redactTestEvent(t *testing.T, r *excelImageRedactor, kind string, content int, text string) [][]byte {
	t.Helper()
	p := map[string]any{"type": kind, "item_id": "item_1", "output_index": 0, "content_index": content}
	if strings.HasSuffix(kind, ".delta") {
		p["delta"] = text
	} else {
		p["text"] = text
	}
	raw, err := json.Marshal(p)
	require.NoError(t, err)
	events, err := r.events(raw)
	require.NoError(t, err)
	return events
}

func TestExcelImageRedactorEveryStreamSplit(t *testing.T) {
	_, link, token := imageRedactorFixture(t)
	for _, sensitive := range []string{link, ExcelBPSImagePath + "?token=" + token, token, strings.ReplaceAll(link, "/", `\/`), url.QueryEscape(link)} {
		for split := 0; split <= len(sensitive); split++ {
			r, _, _ := imageRedactorFixture(t)
			var events [][]byte
			events = append(events, redactTestEvent(t, r, "response.output_text.delta", 0, "图片："+sensitive[:split])...)
			events = append(events, redactTestEvent(t, r, "response.output_text.delta", 0, sensitive[split:]+" end.")...)
			events = append(events, redactTestEvent(t, r, "response.output_text.done", 0, "图片："+sensitive+" end.")...)
			var delta strings.Builder
			for i, event := range events {
				require.Equal(t, int64(i), gjson.GetBytes(event, "sequence_number").Int())
				require.NotContains(t, string(event), token)
				require.NotContains(t, string(event), ExcelBPSImagePath)
				delta.WriteString(gjson.GetBytes(event, "delta").String())
			}
			require.Equal(t, "图片："+excelImageReference+" end.", delta.String(), "split=%d sensitive=%s", split, sensitive)
		}
	}
}

func TestExcelImageRedactorInterleavedPartsAndCharacterDeltas(t *testing.T) {
	r, link, token := imageRedactorFixture(t)
	var parts [2]strings.Builder
	collect := func(events [][]byte) {
		for _, event := range events {
			require.NotContains(t, string(event), token)
			if gjson.GetBytes(event, "type").String() == "response.output_text.delta" {
				parts[gjson.GetBytes(event, "content_index").Int()].WriteString(gjson.GetBytes(event, "delta").String())
			}
		}
	}
	collect(redactTestEvent(t, r, "response.output_text.delta", 1, "https://images.example/v1/"))
	collect(redactTestEvent(t, r, "response.output_text.delta", 0, "unrelated https://public.example/picture.png"))
	collect(redactTestEvent(t, r, "response.output_text.done", 0, "unrelated https://public.example/picture.png"))
	for _, c := range strings.TrimPrefix(link, "https://images.example/v1/") {
		collect(redactTestEvent(t, r, "response.output_text.delta", 1, string(c)))
	}
	collect(redactTestEvent(t, r, "response.output_text.done", 1, link))
	require.Equal(t, "unrelated https://public.example/picture.png", parts[0].String())
	require.Equal(t, excelImageReference, parts[1].String())
}

func TestExcelImageRedactorTerminalToolsErrorsAndNumbers(t *testing.T) {
	r, link, token := imageRedactorFixture(t)
	raw := []byte(fmt.Sprintf(`{"type":"response.completed","response":{"output":[{"arguments":"{\"image\":\"%s\",\"number\":9007199254740993}","url":"%s"}],"error":{"message":"%s"},"usage":{"input_tokens":9007199254740993}}}`, link, link, token))
	events, err := r.events(raw)
	require.NoError(t, err)
	require.Len(t, events, 1)
	require.NotContains(t, string(events[0]), token)
	require.NotContains(t, string(events[0]), "images.example")
	require.Contains(t, string(events[0]), "9007199254740993")
	require.JSONEq(t, `{"image":"[uploaded image]","number":9007199254740993}`, gjson.GetBytes(events[0], "response.output.0.arguments").String())
	require.NotContains(t, string(r.redactJSON([]byte(fmt.Sprintf(`{"error":{"message":"%s"}}`, link)))), "images.example")
	// Escaped slash and Unicode escapes must be decoded before replacement.
	escaped := strings.ReplaceAll(string(raw), "/", `\u002f`)
	events, err = r.events([]byte(escaped))
	require.NoError(t, err)
	require.NotContains(t, string(events[0]), token)
	// A failed/truncated answer cannot release an almost-complete capability.
	r, _, _ = imageRedactorFixture(t)
	redactTestEvent(t, r, "response.output_text.delta", 0, token[:63])
	events, err = r.events([]byte(`{"type":"response.failed","response":{"status":"failed"}}`))
	require.NoError(t, err)
	require.Equal(t, excelImageReference, gjson.GetBytes(events[0], "delta").String())
}

func TestExcelImageRedactorPreservesOrdinaryToolBytes(t *testing.T) {
	r, _, _ := imageRedactorFixture(t)
	args := "{\"x\": 9007199254740993, \"script\": \"if (x < 2) { x++; }\\n\"}"
	raw, _ := json.Marshal(map[string]any{"type": "response.function_call_arguments.delta", "delta": args})
	events, err := r.events(raw)
	require.NoError(t, err)
	var out strings.Builder
	for _, event := range events {
		out.WriteString(gjson.GetBytes(event, "delta").String())
	}
	events = redactTestEvent(t, r, "response.function_call_arguments.done", 0, "")
	for _, event := range events {
		out.WriteString(gjson.GetBytes(event, "delta").String())
	}
	require.Equal(t, args, out.String())
	require.Nil(t, newExcelImageRedactor([]byte(`{"image_url":"https://public.example/image.png"}`)))
}

func TestExcelBPSForwardNeverReturnsImageCapabilities(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprint(stream), func(t *testing.T) {
			storage := imageTestService(t)
			storage.frontendURL = "https://images.example"
			body := inlineTestBody(t, inlineTestImage(t, 1))
			_, plan, err := prepareExcelBPSImages(body)
			require.NoError(t, err)
			token := storage.imageToken("account:300/key:0/thread:", plan.images[0].data)
			link := storage.frontendURL + ExcelBPSImagePath + "?token=" + token
			var wire bytes.Buffer
			emit := func(p any) { raw, _ := json.Marshal(p); fmt.Fprintf(&wire, "data: %s\n\n", raw) }
			for _, c := range link {
				emit(map[string]any{"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": string(c)})
			}
			emit(map[string]any{"type": "response.output_text.done", "item_id": "msg_1", "output_index": 0, "content_index": 0, "text": link})
			emit(map[string]any{"type": "response.completed", "response": map[string]any{"id": "resp_image", "status": "completed", "model": "gpt-6-astra", "output": []any{map[string]any{"type": "message", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": link}}}}, "usage": map[string]any{"input_tokens": 10, "output_tokens": 2}}})
			upstream := &httpUpstreamRecorder{resp: excelBPSTestResponse(wire.String())}
			svc := openAIClientToolsTestService(upstream)
			svc.excelBPSImages = storage
			c, rec := imageGatewayContext()
			body, err = sjson.SetBytes(body, "stream", stream)
			require.NoError(t, err)
			result, err := svc.Forward(context.Background(), c, excelAccount(), body)
			require.NoError(t, err)
			require.Equal(t, 10, result.Usage.InputTokens)
			require.Contains(t, string(upstream.lastBody), link, "upstream still receives the original stable URL")
			require.NotContains(t, rec.Body.String(), token)
			require.NotContains(t, rec.Body.String(), "images.example")
			require.NotContains(t, rec.Body.String(), ExcelBPSImagePath)
			require.Contains(t, rec.Body.String(), excelImageReference)
		})
	}
}
