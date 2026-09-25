package service

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func inlineTestImage(t *testing.T, shade uint8) string {
	t.Helper()
	var b bytes.Buffer
	im := image.NewRGBA(image.Rect(0, 0, 2, 2))
	im.Set(0, 0, color.RGBA{shade, 22, 33, 255})
	require.NoError(t, png.Encode(&b, im))
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(b.Bytes())
}
func inlineTestBody(t *testing.T, images ...string) []byte {
	t.Helper()
	content := []any{map[string]any{"type": "input_text", "text": "Look at these images"}}
	for _, im := range images {
		content = append(content, map[string]any{"type": "input_image", "image_url": im, "detail": "high"})
	}
	body, err := json.Marshal(map[string]any{"model": "gpt-6-astra", "input": []any{map[string]any{"role": "user", "content": content}}})
	require.NoError(t, err)
	return body
}

func TestExcelBPSImagesValidateBeforeUpload(t *testing.T) {
	valid := inlineTestImage(t, 1)
	for name, raw := range map[string]string{"malformed": "data:image/png;base64,not-base64", "wrong-mime": strings.Replace(valid, "image/png", "image/jpeg", 1), "svg": "data:image/svg+xml;base64,PHN2Zz4=", "empty": "data:image/png;base64,", "oversized": "data:image/png;base64," + strings.Repeat("A", base64.StdEncoding.EncodedLen(excelBPSMaxImageBytes)+1)} {
		t.Run(name, func(t *testing.T) {
			_, _, err := prepareExcelBPSImages(inlineTestBody(t, raw))
			require.Error(t, err)
			require.NotContains(t, err.Error(), raw)
		})
	}
	tooMany := make([]string, 17)
	for i := range tooMany {
		tooMany[i] = valid
	}
	_, _, err := prepareExcelBPSImages(inlineTestBody(t, tooMany...))
	require.ErrorContains(t, err, "at most 16")
}

func TestExcelBPSImagesPreserveOpaqueFieldsAndNumbers(t *testing.T) {
	inline := inlineTestImage(t, 2)
	body := []byte(fmt.Sprintf(`{"model":"gpt-6-astra","metadata":{"number":9007199254740993},"input":[{"type":"function_call","call_id":"call_1","name":"read","arguments":"{\"image_url\":\"%s\"}"},{"type":"function_call_output","call_id":"call_1","output":[{"type":"input_image","image_url":"%s"}]}]}`, inline, inline))
	prepared, plan, err := prepareExcelBPSImages(body)
	require.NoError(t, err)
	require.Len(t, plan.images, 1)
	require.Equal(t, gjson.GetBytes(body, "input.0.arguments").String(), gjson.GetBytes(prepared, "input.0.arguments").String())
	require.Contains(t, string(prepared), "9007199254740993")
	require.Contains(t, gjson.GetBytes(prepared, "input.1.output.0.image_url").String(), "https://inline-image.invalid/")
	urlBody := inlineTestBody(t, "https://example.com/private.png?signature=unchanged")
	got, plan, err := prepareExcelBPSImages(urlBody)
	require.NoError(t, err)
	require.Empty(t, plan.images)
	require.Equal(t, urlBody, got)
}

func imageGatewayContext() (*gin.Context, *httptest.ResponseRecorder) {
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest("POST", "/v1/responses", nil)
	return c, rec
}

func TestExcelBPSImagesRedactsSignedLinksFromDiagnostics(t *testing.T) {
	raw := `{"error":{"message":"Could not fetch https://store.example/object?X-Amz-Credential=secret-credential&X-Amz-Signature=secret-signature&X-Amz-Security-Token=secret-session"}}`
	clean := excelBPSSanitizeErrorBody(raw, "account-token", excelAccount())
	for _, secret := range []string{"secret-credential", "secret-signature", "secret-session"} {
		require.NotContains(t, clean, secret)
	}
}
