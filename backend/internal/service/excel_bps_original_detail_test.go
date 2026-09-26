package service

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

func TestExcelBPSOriginalDetailSurvivesUploadAndCache(t *testing.T) {
	up := &attachmentUpstream{t: t}
	svc := attachmentTestGateway(up)
	inline := inlineTestImage(t, 1)
	body := inlineTestBody(t, inline, inline)
	require.NoError(t, excelBPSImageParts(body, func(path string, _ gjson.Result) error {
		var err error
		body, err = sjson.SetBytes(body, strings.TrimSuffix(path, "image_url")+"detail", "original")
		return err
	}))
	for i := 0; i < 3; i++ {
		c, rec := imageGatewayContext()
		c.Set("api_key", &APIKey{ID: 1})
		_, err := svc.Forward(context.Background(), c, excelAccount(), body)
		require.NoError(t, err)
		require.Equal(t, 200, rec.Code)
	}
	require.Equal(t, 1, up.uploads)
	require.Equal(t, 3, up.models)
	for _, wire := range up.modelBodies {
		require.Equal(t, up.modelBodies[0], wire)
		count := 0
		require.NoError(t, excelBPSImageParts(wire, func(_ string, part gjson.Result) error {
			require.Equal(t, "original", part.Get("detail").String())
			require.True(t, validExcelBPSFileID(part.Get("file_id").String()))
			count++
			return nil
		}))
		require.Equal(t, 2, count)
	}
}
