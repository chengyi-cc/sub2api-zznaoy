package basispoints

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestHistoryMessageIDs(t *testing.T) {
	for _, explicitType := range []bool{true, false} {
		for _, tc := range []struct {
			name      string
			id        string
			omitID    bool
			normalize bool
		}{
			{name: "codex_item", id: "item_fbc3f9da85635aed626cfb58", normalize: true},
			{name: "native_message", id: "msg_existing"},
			{name: "maximum_length", id: "msg_" + strings.Repeat("a", 60)},
			{name: "overlong_message", id: "msg_" + strings.Repeat("a", 61), normalize: true},
			{name: "missing", omitID: true},
			{name: "empty"},
		} {
			t.Run(fmt.Sprintf("%s/explicit_type=%t", tc.name, explicitType), func(t *testing.T) {
				original := object{
					"role": "assistant", "status": "completed", "phase": "final_answer",
					"content": []any{object{"type": "output_text", "text": "Original answer\nunchanged.", "annotations": []any{}}},
				}
				if explicitType {
					original["type"] = "message"
				}
				if !tc.omitID {
					original["id"] = tc.id
				}
				source := testSource()
				source["input"] = []any{original}
				var normalizedID string
				for attempt := 0; attempt < 2; attempt++ {
					body, _ := mustPrepare(t, source, "scope", new(ReplayCache))
					items := mustTestValue[[]any](t, body["input"])
					got := mustTestValue[object](t, items[len(items)-1])
					id := text(got["id"])
					if tc.normalize {
						if !strings.HasPrefix(id, "msg_") || len(id) > 64 || id == tc.id {
							t.Fatalf("history message still has an invalid upstream ID: %q", id)
						}
					} else if id != tc.id {
						t.Fatalf("compatible message ID changed: got %q, want %q", id, tc.id)
					}
					if _, exists := got["id"]; exists == tc.omitID {
						t.Fatal("message ID presence changed")
					}
					if attempt > 0 && id != normalizedID {
						t.Fatal("message ID changed after replay cache loss")
					}
					normalizedID = id
					want := make(object, len(original))
					for key, value := range original {
						want[key] = value
					}
					if tc.normalize {
						want["id"] = id
					}
					if !reflect.DeepEqual(got, want) {
						t.Fatalf("message fields changed beyond ID: got %#v, want %#v", got, want)
					}
				}
				if text(original["id"]) != tc.id {
					t.Fatal("caller history was modified")
				}
				if tc.normalize {
					original["id"] = normalizedID
					body, _ := mustPrepare(t, source, "scope", nil)
					items := mustTestValue[[]any](t, body["input"])
					if text(mustTestValue[object](t, items[len(items)-1])["id"]) != normalizedID {
						t.Fatal("normalized message ID changed when replayed")
					}
				}
			})
		}
	}
}

func TestHistoryMessageIDLongConversationRegression(t *testing.T) {
	input := make([]any, 0, 462)
	for i := 0; i < 460; i++ {
		item := message("user", fmt.Sprintf("Earlier message %d", i))
		item["id"] = fmt.Sprintf("msg_history_%d", i)
		input = append(input, item)
	}
	legacy := message("assistant", "Completed the previous task.")
	legacy["id"] = "item_fbc3f9da85635aed626cfb58"
	input = append(input, legacy, message("user", "Continue."))
	source := testSource()
	source["input"] = input
	body, _ := mustPrepare(t, source, "scope", nil)
	items := mustTestValue[[]any](t, body["input"])
	if len(items) != len(input)+2 {
		t.Fatalf("history items were lost or added: got %d", len(items))
	}
	for i, raw := range input {
		got := mustTestValue[object](t, items[i+2])
		if i == 460 {
			if !strings.HasPrefix(text(got["id"]), "msg_") {
				t.Fatalf("input[462].id still rejected by BPS: %v", got["id"])
			}
			got["id"] = legacy["id"]
		}
		if !reflect.DeepEqual(got, raw) {
			t.Fatalf("history content or ordering changed at input %d", i)
		}
	}
}
