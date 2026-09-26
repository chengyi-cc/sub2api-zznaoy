package service

import (
	"encoding/json"
	"regexp"
	"strconv"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

var excelBPSEncryptedRejectionMarker = regexp.MustCompile(`The encrypted content ([A-Za-z0-9_+./=-]{1,256}) could not be verified`)
var excelBPSEncryptedRejectionPath = regexp.MustCompile(`^input\[([0-9]+)\]\.encrypted_content$`)

// Recovery is limited to optional reasoning with retained plaintext history.
// Compaction may be the only copy of the conversation: never discard it.
// Unknown encrypted blocks and ambiguous upstream identifiers fail closed.
func excelBPSReasoningRecoveryInput(body []byte) ([]gjson.Result, bool) {
	input := gjson.GetBytes(body, "input")
	if !input.IsArray() {
		return nil, false
	}
	items := input.Array()
	hasUserText := false
	for _, item := range items {
		kind := item.Get("type").String()
		if kind == "compaction" || kind == "compaction_summary" || kind == "item_reference" {
			return nil, false
		}
		if (kind == "message" || kind == "") && item.Get("role").String() == "user" {
			content := item.Get("content")
			if content.Type == gjson.String && strings.TrimSpace(content.String()) != "" {
				hasUserText = true
			}
			content.ForEach(func(_, part gjson.Result) bool {
				if part.Get("type").String() == "input_text" && strings.TrimSpace(part.Get("text").String()) != "" {
					hasUserText = true
				}
				return true
			})
		}
		if kind == "reasoning" {
			// No other part of this item may contain encrypted user/agent data.
			unsafe := false
			item.ForEach(func(key, value gjson.Result) bool {
				if key.String() != "encrypted_content" && excelBPSHasEncryptedField(value) {
					unsafe = true
				}
				return !unsafe
			})
			if unsafe {
				return nil, false
			}
		} else if excelBPSHasEncryptedField(item) {
			return nil, false
		}
	}
	return items, hasUserText
}

func excelBPSHasEncryptedField(value gjson.Result) bool {
	if !value.IsObject() && !value.IsArray() {
		return false
	}
	found := false
	value.ForEach(func(key, child gjson.Result) bool {
		if key.String() == "encrypted_content" || excelBPSHasEncryptedField(child) {
			found = true
		}
		return !found
	})
	return found
}

func excelBPSDropRejectedReasoning(body []byte, invalid map[string]struct{}) ([]byte, int) {
	if len(invalid) == 0 {
		return body, 0
	}
	items, safe := excelBPSReasoningRecoveryInput(body)
	if !safe {
		return body, 0
	}
	kept := make([]json.RawMessage, 0, len(items))
	dropped := 0
	for _, item := range items {
		cipher := item.Get("encrypted_content")
		_, matched := invalid[openAIEncryptedContentDigest(cipher.String())]
		if item.Get("type").String() == "reasoning" && cipher.Type == gjson.String && cipher.String() != "" && matched {
			dropped++
			continue
		}
		kept = append(kept, json.RawMessage(item.Raw))
	}
	if dropped == 0 {
		return body, 0
	}
	raw, err := json.Marshal(kept)
	if err != nil {
		return body, 0
	}
	rebuilt, err := sjson.SetRawBytes(body, "input", raw)
	if err != nil {
		return body, 0
	}
	return rebuilt, dropped
}

func excelBPSRejectedReasoningRetry(body, rejection []byte) ([]byte, []string) {
	if !gjson.ValidBytes(rejection) || gjson.GetBytes(rejection, "error.code").String() != "invalid_encrypted_content" {
		return body, nil
	}
	items, safe := excelBPSReasoningRecoveryInput(body)
	if !safe {
		return body, nil
	}
	position := -1
	param := gjson.GetBytes(rejection, "error.param").String()
	if param != "" {
		match := excelBPSEncryptedRejectionPath.FindStringSubmatch(param)
		if len(match) != 2 {
			return body, nil
		}
		var err error
		position, err = strconv.Atoi(match[1])
		if err != nil || position >= len(items) {
			return body, nil
		}
	}
	marker := excelBPSEncryptedRejectionMarker.FindStringSubmatch(gjson.GetBytes(rejection, "error.message").String())
	if position < 0 && len(marker) != 2 {
		return body, nil
	}
	matched := make(map[string]struct{})
	for index, item := range items {
		cipher := item.Get("encrypted_content")
		if item.Get("type").String() != "reasoning" || cipher.Type != gjson.String || cipher.String() == "" {
			continue
		}
		if position >= 0 && index != position {
			continue
		}
		if len(marker) == 2 && !excelBPSCipherMatchesMarker(cipher.String(), marker[1]) {
			continue
		}
		matched[openAIEncryptedContentDigest(cipher.String())] = struct{}{}
	}
	if len(matched) != 1 {
		return body, nil
	}
	rebuilt, removed := excelBPSDropRejectedReasoning(body, matched)
	if removed == 0 {
		return body, nil
	}
	for digest := range matched {
		return rebuilt, []string{digest}
	}
	return body, nil
}

func excelBPSCipherMatchesMarker(cipher, marker string) bool {
	if !strings.Contains(marker, "...") {
		return cipher == marker
	}
	parts := strings.Split(marker, "...")
	return len(parts) == 2 && len(parts[0]) >= 4 && len(parts[1]) >= 4 &&
		len(cipher) >= len(parts[0])+len(parts[1]) && strings.HasPrefix(cipher, parts[0]) && strings.HasSuffix(cipher, parts[1])
}

// Diagnostics contain only protocol categories and counts, never user content,
// ciphertext, tool arguments, credentials or caller-controlled type strings.
func excelBPSEncryptedHistoryDiagnostic(body []byte) string {
	counts := map[string]int{}
	gjson.GetBytes(body, "input").ForEach(func(_, item gjson.Result) bool {
		if item.Get("encrypted_content").Exists() {
			switch item.Get("type").String() {
			case "reasoning", "compaction", "compaction_summary":
				counts[item.Get("type").String()]++
			default:
				counts["other_top_level"]++
			}
		}
		if excelBPSHasEncryptedField(item.Get("content")) {
			counts["encrypted_message_content"]++
		}
		return true
	})
	raw, _ := json.Marshal(counts)
	return string(raw)
}
