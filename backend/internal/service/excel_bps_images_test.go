package service

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/service/basispoints"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

type temporaryImageFake struct {
	keys, deleted []string
	ttls          []time.Duration
	data          [][]byte
	failAt        int
	badURL        bool
	cutoff        chan time.Time
}

func (f *temporaryImageFake) Save(context.Context, string, string, []byte) (string, error) {
	panic("must not use public image storage")
}
func (f *temporaryImageFake) SaveTemporary(_ context.Context, key, mime string, data []byte, ttl time.Duration) (string, error) {
	f.keys = append(f.keys, key)
	f.ttls = append(f.ttls, ttl)
	f.data = append(f.data, bytes.Clone(data))
	if len(f.keys) == f.failAt {
		return "", errors.New("secret-storage-error")
	}
	if f.badURL {
		return "http://private.invalid/object", nil
	}
	return "https://private.example/" + key + "?X-Amz-Expires=300&signature=test", nil
}
func (f *temporaryImageFake) DeleteTemporary(_ context.Context, key string) error {
	f.deleted = append(f.deleted, key)
	return nil
}
func (f *temporaryImageFake) DeleteExpiredTemporary(_ context.Context, before time.Time) error {
	if f.cutoff != nil {
		f.cutoff <- before
	}
	return nil
}
func imageTestService(f *temporaryImageFake) *ExcelBPSImageService {
	s := NewExcelBPSImageService(nil)
	s.resolve = func(context.Context, bool) (TemporaryImageStorage, error) { return f, nil }
	return s
}
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

func TestExcelBPSImagesUploadAndRefreshHistory(t *testing.T) {
	inline := inlineTestImage(t, 1)
	body := inlineTestBody(t, inline, inline)
	original := bytes.Clone(body)
	f := &temporaryImageFake{}
	s := imageTestService(f)
	prepared, plan, err := prepareExcelBPSImages(body)
	require.NoError(t, err)
	require.Len(t, plan.images, 1)
	wire, _, err := basispoints.Prepare(prepared, "scope", nil)
	require.NoError(t, err)
	first, err := s.upload(context.Background(), wire, plan)
	require.NoError(t, err)
	require.Len(t, f.keys, 1)
	require.Equal(t, 5*time.Minute, f.ttls[0])
	require.NotContains(t, string(first), "data:image")
	require.NotContains(t, string(first), "inline-image.invalid")
	require.Contains(t, string(first), "X-Amz-Expires=300")
	require.Equal(t, original, body)
	require.Contains(t, string(first), `"detail":"high"`)
	// A later request with original history obtains new object keys/links. No
	// client history is rewritten to contain a soon-to-expire storage URL.
	second, err := s.upload(context.Background(), wire, plan)
	require.NoError(t, err)
	require.NotEqual(t, f.keys[0], f.keys[1])
	require.NotEqual(t, string(first), string(second))
	require.Equal(t, gjson.GetBytes(first, "metadata.turn_id").String(), gjson.GetBytes(second, "metadata.turn_id").String())
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

func TestExcelBPSImagesRollbackUploadFailure(t *testing.T) {
	prepared, plan, err := prepareExcelBPSImages(inlineTestBody(t, inlineTestImage(t, 1), inlineTestImage(t, 2)))
	require.NoError(t, err)
	for _, f := range []*temporaryImageFake{{failAt: 2}, {badURL: true}} {
		_, err := imageTestService(f).upload(context.Background(), prepared, plan)
		require.Error(t, err)
		require.Equal(t, f.keys, f.deleted)
		require.NotContains(t, err.Error(), "secret-storage-error")
	}
}

func TestExcelBPSImagesGatewayRoutingAndErrors(t *testing.T) {
	for _, tc := range []struct {
		name             string
		enabled, storage bool
		status           int
	}{{"Excel upload", true, true, 200}, {"Excel missing storage", true, false, 503}, {"native unchanged", false, false, 200}} {
		t.Run(tc.name, func(t *testing.T) {
			f := &temporaryImageFake{}
			upstream := &httpUpstreamRecorder{resp: excelBPSTestResponse("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_image\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Image received\"}]}],\"usage\":{\"input_tokens\":10,\"output_tokens\":2}}}\n\n")}
			svc := openAIClientToolsTestService(upstream)
			if tc.storage {
				svc.excelBPSImages = imageTestService(f)
			}
			a := excelAccount()
			a.Extra["openai_excel_bps"] = tc.enabled
			c, rec := imageGatewayContext()
			_, err := svc.Forward(context.Background(), c, a, inlineTestBody(t, inlineTestImage(t, 1)))
			require.Equal(t, tc.status, rec.Code)
			if tc.status != 200 {
				require.Error(t, err)
				require.Empty(t, upstream.requests)
				require.True(t, IsResponseCommitted(c))
				return
			}
			require.NoError(t, err)
			if tc.enabled {
				require.Len(t, f.keys, 1)
				require.NotContains(t, string(upstream.lastBody), "data:image")
				require.Contains(t, string(upstream.lastBody), "X-Amz-Expires=300")
			} else {
				require.Contains(t, string(upstream.lastBody), "data:image/png;base64,")
				require.Empty(t, f.keys)
			}
		})
	}
	// Unsupported protocol requests must fail before any upload side effects.
	f := &temporaryImageFake{}
	svc := openAIClientToolsTestService(&httpUpstreamRecorder{})
	svc.excelBPSImages = imageTestService(f)
	body := inlineTestBody(t, inlineTestImage(t, 1))
	body = bytes.Replace(body, []byte(`"model":`), []byte(`"previous_response_id":"old","model":`), 1)
	c, _ := imageGatewayContext()
	_, err := svc.Forward(context.Background(), c, excelAccount(), body)
	require.Error(t, err)
	require.Empty(t, f.keys)
}
func imageGatewayContext() (*gin.Context, *httptest.ResponseRecorder) {
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest("POST", "/v1/responses", nil)
	return c, rec
}

func TestExcelBPSImagesCleanupStartsWithoutNewUploads(t *testing.T) {
	f := &temporaryImageFake{cutoff: make(chan time.Time, 1)}
	s := imageTestService(f)
	s.Start()
	defer s.Stop()
	select {
	case cutoff := <-f.cutoff:
		require.WithinDuration(t, time.Now().Add(-6*time.Minute), cutoff, time.Second)
	case <-time.After(2 * time.Second):
		t.Fatal("restart cleanup did not run")
	}
}

func TestExcelBPSImagesStorageSettingsUsePrivateLinks(t *testing.T) {
	for _, enabled := range []bool{true, false} {
		f := &temporaryImageFake{}
		var received config.ImageStorageConfig
		settings := NewImageStorageSettingService(nil, nil, nil, func(_ context.Context, cfg *config.ImageStorageConfig) (ImageStorage, error) {
			received = *cfg
			return f, nil
		}, config.ImageStorageConfig{Enabled: enabled, Bucket: "images", AccessKeyID: "key", SecretAccessKey: "secret", PublicBaseURL: "https://cdn.example", PresignExpiry: 24})
		s := NewExcelBPSImageService(settings)
		_, err := s.resolveStorage(context.Background(), true)
		if enabled {
			require.NoError(t, err)
			require.Empty(t, received.PublicBaseURL)
		} else {
			require.Error(t, err)
		}
		// Disabling new uploads does not disable cleanup of earlier objects.
		_, err = s.resolveStorage(context.Background(), false)
		require.NoError(t, err)
	}
}

func TestExcelBPSImagesRedactsSignedLinksFromDiagnostics(t *testing.T) {
	raw := `{"error":{"message":"Could not fetch https://store.example/object?X-Amz-Credential=secret-credential&X-Amz-Signature=secret-signature&X-Amz-Security-Token=secret-session"}}`
	clean := excelBPSSanitizeErrorBody(raw, "account-token", excelAccount())
	for _, secret := range []string{"secret-credential", "secret-signature", "secret-session"} {
		require.NotContains(t, clean, secret)
	}
}
