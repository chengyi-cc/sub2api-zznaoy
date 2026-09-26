package service

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

func toolImageCacheTestBody(t *testing.T, kind string) []byte {
	t.Helper()
	call := map[string]any{"type": kind + "_call", "name": "view_image", "call_id": "call_image_cache"}
	if kind == "custom_tool" {
		call["input"] = "recorded image read"
	} else {
		call["arguments"] = "{}"
	}
	body, err := json.Marshal(map[string]any{
		"model": "gpt-6-astra", "prompt_cache_key": "tool-image-cache-test",
		"input": []any{
			map[string]any{"role": "user", "content": "Inspect this recorded image."},
			call,
			map[string]any{"type": kind + "_call_output", "call_id": "call_image_cache", "output": []any{
				map[string]any{"type": "input_text", "text": "Image tool result"},
				map[string]any{"type": "input_image", "image_url": inlineTestImage(t, 9), "detail": "high"},
			}},
		},
	})
	require.NoError(t, err)
	return body
}

func forwardToolImageCacheTest(t *testing.T, svc *OpenAIGatewayService, account *Account, body []byte) {
	t.Helper()
	c, rec := imageGatewayContext()
	c.Set("api_key", &APIKey{ID: 1})
	_, err := svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.Equal(t, 200, rec.Code)
}

func TestExcelBPSToolImagesCacheStableRepeatsAndAppendedHistory(t *testing.T) {
	for _, kind := range []string{"function", "custom_tool"} {
		t.Run(kind, func(t *testing.T) {
			up := &attachmentUpstream{t: t}
			svc, account := attachmentTestGateway(up), excelAccount()
			body := toolImageCacheTestBody(t, kind)
			original := bytes.Clone(body)
			for i := 0; i < 5; i++ {
				forwardToolImageCacheTest(t, svc, account, body)
				require.Equal(t, up.modelBodies[0], up.modelBodies[i], "identical requests must produce identical upstream bytes")
			}
			require.Equal(t, 1, up.uploads, "replayed image history must reuse the uploaded attachment")
			require.Equal(t, original, body)
			first := up.modelBodies[0]
			cacheKey := gjson.GetBytes(first, "prompt_cache_key").String()
			require.NotEmpty(t, cacheKey)
			oldItems := gjson.GetBytes(first, "input").Array()
			for _, followup := range []map[string]any{
				{"type": "message", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": "Recorded image inspected."}}},
				{"type": "message", "role": "user", "content": "Continue."},
			} {
				var err error
				body, err = sjson.SetBytes(body, "input.-1", followup)
				require.NoError(t, err)
				forwardToolImageCacheTest(t, svc, account, body)
				current := up.modelBodies[len(up.modelBodies)-1]
				require.Equal(t, cacheKey, gjson.GetBytes(current, "prompt_cache_key").String())
				require.Equal(t, gjson.GetBytes(first, "metadata.task_id").String(), gjson.GetBytes(current, "metadata.task_id").String())
				items := gjson.GetBytes(current, "input").Array()
				require.Greater(t, len(items), len(oldItems))
				for i, old := range oldItems {
					require.Equal(t, old.Raw, items[i].Raw, "appending a turn must preserve every previous image-history item")
				}
			}
			require.Equal(t, 1, up.uploads)
		})
	}
}

func TestExcelBPSToolImagesCacheRestartAndExpiredAttachment(t *testing.T) {
	up := &attachmentUpstream{t: t}
	svc, account := attachmentTestGateway(up), excelAccount()
	body := toolImageCacheTestBody(t, "custom_tool")
	forwardToolImageCacheTest(t, svc, account, body)
	first := up.modelBodies[0]
	// The attachment cache is process-local. Recreating it necessarily uploads
	// again; the new file ID must not change any task/turn/routing identity.
	svc.excelBPSImages = NewExcelBPSImageService(nil)
	forwardToolImageCacheTest(t, svc, account, body)
	require.Equal(t, 2, up.uploads)
	second := up.modelBodies[1]
	require.NotEqual(t, first, second)
	require.Equal(t, first, bytes.ReplaceAll(second, []byte("file-test-2"), []byte("file-test-1")))
	scope := excelBPSAttachmentScope{account.ID, 1, "test-account"}
	svc.excelBPSImages.invalidateRejected(scope, second, []byte(`{"error":{"code":"file_expired"}}`))
	forwardToolImageCacheTest(t, svc, account, body)
	require.Equal(t, 3, up.uploads)
	third := up.modelBodies[2]
	require.Equal(t, first, bytes.ReplaceAll(third, []byte("file-test-3"), []byte("file-test-1")))
	forwardToolImageCacheTest(t, svc, account, body)
	require.Equal(t, third, up.modelBodies[3])
	require.Equal(t, 3, up.uploads)
}
