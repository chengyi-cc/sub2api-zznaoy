package service

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"image"
	_ "image/gif"
	"net/http"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	_ "golang.org/x/image/webp"
)

const excelBPSMaxImageBytes = 20 << 20
const excelBPSMaxTotalImageBytes = 32 << 20
const excelBPSMaxImages = 16

type excelBPSInlineImage struct {
	placeholder, mime string
	data              []byte
}
type excelBPSImagePlan struct{ images []excelBPSInlineImage }

// Visit actual media parts only. Text, tool schemas and JSON tool arguments
// containing image-like keys must remain opaque and byte-for-byte unchanged.
func excelBPSImageParts(body []byte, visit func(string, gjson.Result) error) error {
	for i, item := range gjson.GetBytes(body, "input").Array() {
		for _, field := range []string{"content", "output"} {
			parts := item.Get(field)
			if !parts.IsArray() {
				continue
			}
			for j, part := range parts.Array() {
				if part.Get("type").String() == "input_image" {
					if err := visit(fmt.Sprintf("input.%d.%s.%d.image_url", i, field, j), part); err != nil {
						return err
					}
				}
			}
		}
	}
	return nil
}

func prepareExcelBPSImages(body []byte) ([]byte, *excelBPSImagePlan, error) {
	plan := &excelBPSImagePlan{}
	seen := make(map[string]bool)
	total, count := 0, 0
	updated := body
	err := excelBPSImageParts(body, func(path string, part gjson.Result) error {
		raw := part.Get("image_url").String()
		if !strings.HasPrefix(strings.ToLower(strings.TrimSpace(raw)), "data:") {
			return nil
		}
		count++
		if count > excelBPSMaxImages {
			return fmt.Errorf("Excel BPS accepts at most 16 inline images per request")
		}
		mime, data, err := decodeExcelBPSImage(raw)
		if err != nil {
			return err
		}
		total += len(data)
		if total > excelBPSMaxTotalImageBytes {
			return fmt.Errorf("Excel BPS inline images exceed 32 MiB in total")
		}
		// Stable placeholders keep turn identities independent of random object
		// keys and expiring signatures. The protocol is validated before upload.
		placeholder := fmt.Sprintf("https://inline-image.invalid/%x", sha256.Sum256(data))
		if !seen[placeholder] {
			plan.images = append(plan.images, excelBPSInlineImage{placeholder, mime, data})
			seen[placeholder] = true
		}
		updated, err = sjson.SetBytes(updated, path, placeholder)
		return err
	})
	return updated, plan, err
}

func decodeExcelBPSImage(raw string) (string, []byte, error) {
	header, encoded, ok := strings.Cut(raw, ",")
	header = strings.ToLower(header)
	if !ok || !strings.HasPrefix(header, "data:image/") || !strings.HasSuffix(header, ";base64") {
		return "", nil, fmt.Errorf("Excel BPS inline images require a base64 PNG, JPEG, GIF or WebP data URL")
	}
	mime := strings.TrimSuffix(strings.TrimPrefix(header, "data:"), ";base64")
	if mime == "image/jpg" {
		mime = "image/jpeg"
	}
	if mime != "image/png" && mime != "image/jpeg" && mime != "image/gif" && mime != "image/webp" {
		return "", nil, fmt.Errorf("Excel BPS supports PNG, JPEG, GIF and WebP images only")
	}
	if len(encoded) > base64.StdEncoding.EncodedLen(excelBPSMaxImageBytes) {
		return "", nil, fmt.Errorf("Excel BPS inline image exceeds 20 MiB")
	}
	data, err := base64.StdEncoding.Strict().DecodeString(encoded)
	if err != nil {
		data, err = base64.RawStdEncoding.Strict().DecodeString(encoded)
	}
	if err != nil || len(data) == 0 || len(data) > excelBPSMaxImageBytes {
		return "", nil, fmt.Errorf("invalid or oversized Excel BPS inline image")
	}
	if http.DetectContentType(data) != mime {
		return "", nil, fmt.Errorf("Excel BPS image content does not match its declared type")
	}
	size, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || size.Width <= 0 || size.Height <= 0 || int64(size.Width)*int64(size.Height) > 40_000_000 {
		return "", nil, fmt.Errorf("invalid Excel BPS image dimensions or more than 40 million pixels")
	}
	return mime, data, nil
}
