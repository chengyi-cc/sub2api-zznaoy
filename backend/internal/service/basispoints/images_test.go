package basispoints

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestHTTPSImagesPreserveURLsAndText(t *testing.T) {
	for _, detail := range []string{"", "auto", "low", "high", "original"} {
		image := object{"type": "input_image", "image_url": "https://images.example/photo.png?signature=unchanged%2Fvalue&expires=123"}
		if detail != "" {
			image["detail"] = detail
		}
		item := object{"type": "message", "role": "user", "content": []any{object{"type": "input_text", "text": "Describe this image."}, image}}
		source := testSource()
		source["input"] = []any{item}
		wire, _ := mustPrepare(t, source, "scope", nil)
		items := mustTestValue[[]any](t, wire["input"])
		if !reflect.DeepEqual(items[len(items)-1], item) {
			t.Fatal("image URL, detail or neighboring text was changed")
		}
	}
}

func TestUnsupportedImageFormsReturnActionableErrors(t *testing.T) {
	for name, image := range map[string]object{
		"base64":          {"image_url": "data:image/png;base64,PRIVATE_IMAGE_BYTES"},
		"http":            {"image_url": "http://images.example/photo.png"},
		"relative":        {"image_url": "/photo.png"},
		"local file":      {"image_url": "file:///private/photo.png"},
		"missing host":    {"image_url": "https:///photo.png"},
		"credentials":     {"image_url": "https://private-secret:password@images.example/photo.png"},
		"URL object":      {"image_url": object{"url": "https://images.example/photo.png"}},
		"invalid file ID": {"file_id": "file-"},
		"mixed file ID":   {"image_url": "https://images.example/photo.png", "file_id": "file-private"},
		"invalid detail":  {"image_url": "https://images.example/photo.png", "detail": "unsupported"},
	} {
		t.Run(name, func(t *testing.T) {
			image["type"] = "input_image"
			source := testSource()
			source["input"] = []any{object{"role": "user", "content": []any{image}}}
			raw, _ := json.Marshal(source)
			_, _, err := Prepare(raw, "", nil)
			if err == nil {
				t.Fatal("unsupported image silently forwarded")
			}
			if strings.Contains(err.Error(), "PRIVATE_IMAGE_BYTES") || strings.Contains(err.Error(), "private-secret") {
				t.Fatal("image data or credentials leaked into the error")
			}
			if name == "base64" && (!strings.Contains(err.Error(), "HTTPS image URL") || !strings.Contains(err.Error(), "disable Basispoints")) {
				t.Fatalf("base64 rejection lacks a remedy: %v", err)
			}
		})
	}
}

func TestNativeImageReferencesValidateBeforeForwarding(t *testing.T) {
	for _, detail := range []string{"auto", "low", "high", "original"} {
		image := object{"type": "input_image", "file_id": "file-abc_123-XYZ", "detail": detail}
		source := testSource()
		source["input"] = []any{message("user", "inspect"), object{"type": "message", "role": "user", "content": []any{image}}}
		wire, _ := mustPrepare(t, source, "scope", nil)
		items := mustTestValue[[]any](t, wire["input"])
		got := mustTestValue[[]any](t, mustTestValue[object](t, items[len(items)-1])["content"])[0]
		if !reflect.DeepEqual(image, got) {
			t.Fatal("attachment ID or image detail changed")
		}
	}
	for _, image := range []object{
		{"file_id": nil}, {"file_id": 42}, {"file_id": ""}, {"file_id": "file-"},
		{"file_id": "file-secret/private"}, {"file_id": " file-secret"},
		{"file_id": "file-" + strings.Repeat("x", 252)},
		{"file_id": "file-valid", "image_url": nil},
		{"file_id": "file-valid", "image_url": ""},
		{"file_id": "file-valid", "detail": "invalid"},
	} {
		if err := validateImage(image); err == nil || strings.Contains(err.Error(), "secret") {
			t.Fatal("invalid attachment was accepted or leaked in an error")
		}
	}
}
