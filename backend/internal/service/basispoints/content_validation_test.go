package basispoints

import (
	"errors"
	"strings"
	"testing"
)

func TestContentValidationExposesSafePath(t *testing.T) {
	for _, tc := range []struct {
		part any
		kind string
	}{
		{object{"type": "encrypted_content", "encrypted_content": "private-payload"}, "encrypted_content"},
		{object{"type": "private-payload", "text": "private-payload"}, "unknown"},
		{true, "non_object"},
	} {
		err := validateHistoryContent([]any{tc.part}, 103, "content")
		var validation *ContentValidationError
		if !errors.As(err, &validation) {
			t.Fatalf("unexpected validation error: %v", err)
		}
		if validation.Path != "input[103].content[0]" || validation.ContentType != tc.kind {
			t.Fatalf("unexpected metadata: %+v", validation)
		}
		if strings.Contains(err.Error(), "private-payload") {
			t.Fatal("private payload leaked")
		}
	}
}
