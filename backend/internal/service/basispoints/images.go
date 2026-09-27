package basispoints

import (
	"fmt"
	"net/url"
	"strings"
)

// Accept HTTPS references or syntactically valid native attachment IDs.
// The authenticated upstream remains responsible for file ownership/access.
func validateImage(part object) error {
	if value, exists := part["file_id"]; exists {
		id, ok := value.(string)
		if !ok || !validImageAttachmentID(id) {
			return fmt.Errorf("basispoints input_image requires a valid file_id")
		}
		if _, exists := part["image_url"]; exists {
			return fmt.Errorf("basispoints input_image requires exactly one image reference")
		}
		return validateImageDetail(part)
	}
	raw, ok := part["image_url"].(string)
	if !ok || raw == "" {
		return fmt.Errorf("basispoints input_image requires an HTTPS image_url or file_id")
	}
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(raw)), "data:") {
		return fmt.Errorf("basispoints does not accept data:image/base64 image input; provide an HTTPS image URL, or disable Basispoints and start a new conversation to send this image")
	}
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Scheme != "https" || parsed.Hostname() == "" || parsed.User != nil || parsed.Opaque != "" || strings.TrimSpace(raw) != raw {
		return fmt.Errorf("basispoints input_image requires an absolute HTTPS image URL without embedded credentials")
	}
	return validateImageDetail(part)
}

// Match the gateway's attachment response validation, without echoing values.
func validImageAttachmentID(id string) bool {
	if !strings.HasPrefix(id, "file-") || len(id) <= 5 || len(id) > 256 {
		return false
	}
	for _, c := range id[5:] {
		if !(c >= 'a' && c <= 'z') && !(c >= 'A' && c <= 'Z') && !(c >= '0' && c <= '9') && c != '-' && c != '_' {
			return false
		}
	}
	return true
}

func validateImageDetail(part object) error {
	if detail, exists := part["detail"]; exists && detail != nil {
		switch text(detail) {
		case "auto", "low", "high", "original":
		default:
			return fmt.Errorf("basispoints image detail must be auto, low, high or original")
		}
	}
	return nil
}
