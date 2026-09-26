package service

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestExcelBPSToolImagesPreserveResultBatchesAndReplay(t *testing.T) {
	wire := []byte(`{"metadata":{"task_id":"task","turn_id":"turn","agent_iteration":"3","number":9007199254740993},"prompt_cache_key":"same-key","input":[
 {"type":"function_call_output","id":"fc_1","call_id":"call_1","output":[{"type":"input_text","text":"before\nunchanged"},{"type":"input_image","file_id":"file-first","detail":"high"},{"type":"input_text","text":"after"},{"type":"input_image","file_id":"file-second","detail":"low"}]},
 {"type":"function_call_output","id":"fc_2","call_id":"call_2","output":[{"type":"input_image","image_url":"https://example.com/image.png?sig=unchanged%2F"},{"type":"input_image","file_id":"file-third"}]},
 {"type":"message","role":"user","content":[{"type":"input_text","text":"Continue."}]},
 {"type":"compaction_trigger"}]}`)
	original := bytes.Clone(wire)
	got, err := normalizeExcelBPSToolOutputImages(wire)
	require.NoError(t, err)
	require.Equal(t, original, wire)
	replayed, err := normalizeExcelBPSToolOutputImages(got)
	require.NoError(t, err)
	require.Equal(t, got, replayed, "already lowered image history must remain byte stable")
	require.Equal(t, gjson.GetBytes(wire, "metadata").Raw, gjson.GetBytes(got, "metadata").Raw)
	require.Equal(t, "same-key", gjson.GetBytes(got, "prompt_cache_key").String())
	items := gjson.GetBytes(got, "input").Array()
	require.Len(t, items, 6)
	for i, id := range []string{"call_1", "call_2"} {
		require.Equal(t, "function_call_output", items[i].Get("type").String())
		require.Equal(t, id, items[i].Get("call_id").String())
		require.Equal(t, "user", items[i+2].Get("role").String())
		require.Contains(t, items[i+2].Get("content.0.text").String(), id)
	}
	require.Equal(t, "before\nunchanged", items[0].Get("output.0.text").String())
	require.Equal(t, "after", items[0].Get("output.2.text").String())
	require.Contains(t, items[0].Get("output.1.text").String(), "image 1")
	require.Contains(t, items[0].Get("output.3.text").String(), "image 2")
	require.Equal(t, "file-first", items[2].Get("content.1.file_id").String())
	require.Equal(t, "high", items[2].Get("content.1.detail").String())
	require.Equal(t, "file-second", items[2].Get("content.2.file_id").String())
	require.Equal(t, "low", items[2].Get("content.2.detail").String())
	require.Equal(t, "file-third", items[3].Get("content.1.file_id").String())
	require.Equal(t, "https://example.com/image.png?sig=unchanged%2F", items[1].Get("output.0.image_url").String())
	require.Equal(t, "Continue.", items[4].Get("content.0.text").String())
	require.Equal(t, "compaction_trigger", items[5].Get("type").String())
}

func TestExcelBPSToolImagesLeaveOtherHistoryUnchanged(t *testing.T) {
	for _, wire := range []string{
		`{"input":[{"type":"function_call_output","call_id":"call_text","output":"input_image file_id quoted text"}]}`,
		`{"input":[{"type":"function_call_output","call_id":"call_url","output":[{"type":"input_image","image_url":"https://example.com/image.png"}]}]}`,
		`{"input":[{"type":"message","role":"user","content":[{"type":"input_image","file_id":"file-existing"}]}]}`,
	} {
		got, err := normalizeExcelBPSToolOutputImages([]byte(wire))
		require.NoError(t, err)
		require.Equal(t, wire, string(got))
	}
}
