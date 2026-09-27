package service

import (
	"bytes"
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

func TestExcelBPSToolImageReferencesWithoutUpload(t *testing.T) {
	for _, kind := range []string{"function", "custom_tool"} {
		for _, path := range []string{"/v1/responses", "/v1/responses/compact"} {
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/stream=%t", kind, path, stream), func(t *testing.T) {
					body := toolImageCacheTestBody(t, kind)
					parts := []any{
						map[string]any{"type": "input_text", "text": "before image"},
						map[string]any{"type": "input_image", "file_id": "file-already-uploaded", "detail": "original"},
						map[string]any{"type": "input_text", "text": "between images"},
						map[string]any{"type": "input_image", "image_url": "https://example.com/image.png?sig=preserved%2F", "detail": "high"},
					}
					var err error
					body, err = sjson.SetBytes(body, "input.2.output", parts)
					require.NoError(t, err)
					body, err = sjson.SetBytes(body, "stream", stream)
					require.NoError(t, err)
					original := bytes.Clone(body)
					up := &attachmentUpstream{t: t}
					svc := attachmentTestGateway(up)
					for attempt := 0; attempt < 2; attempt++ {
						c, rec := imageGatewayContext()
						c.Request.URL.Path = path
						c.Set("api_key", &APIKey{ID: 1})
						_, err := svc.Forward(context.Background(), c, excelAccount(), body)
						require.NoError(t, err)
						require.Equal(t, 200, rec.Code)
					}
					require.Equal(t, original, body)
					require.Zero(t, up.uploads, "existing references need no attachment upload")
					require.Equal(t, up.modelBodies[0], up.modelBodies[1], "replays must remain deterministic")
					items := gjson.GetBytes(up.modelBodies[0], "input").Array()
					results := 0
					for i, item := range items {
						if item.Get("type").String() != "function_call_output" {
							continue
						}
						results++
						require.Equal(t, "before image", item.Get("output.0.text").String())
						require.Equal(t, "between images", item.Get("output.2.text").String())
						for _, part := range item.Get("output").Array() {
							require.NotEqual(t, "input_image", part.Get("type").String())
						}
						require.Less(t, i+1, len(items))
						images := items[i+1]
						require.Equal(t, "user", images.Get("role").String())
						require.Contains(t, images.Get("content.0.text").String(), "call_image_cache")
						require.Equal(t, "file-already-uploaded", images.Get("content.1.file_id").String())
						require.Equal(t, "original", images.Get("content.1.detail").String())
						require.Equal(t, "https://example.com/image.png?sig=preserved%2F", images.Get("content.2.image_url").String())
						require.Equal(t, "high", images.Get("content.2.detail").String())
					}
					require.Equal(t, 1, results)
				})
			}
		}
	}
}
