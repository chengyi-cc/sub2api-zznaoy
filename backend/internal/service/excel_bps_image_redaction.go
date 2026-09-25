package service

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/url"
	"reflect"
	"regexp"
	"sort"
	"strings"
)

const excelImageReference = "[uploaded image]"

var excelImageURLPattern = regexp.MustCompile(`https?://[^\s"'<>\\]+`)

type excelImageDelta struct {
	pending  string
	template map[string]any
}

// Redact only capabilities actually attached to this request. The upstream
// request and bridge replay remain untouched; this is the client-facing boundary.
type excelImageRedactor struct {
	patterns []string
	streams  map[string]*excelImageDelta
	order    []string
	sequence int
}

func newExcelImageRedactor(wire []byte) *excelImageRedactor {
	patterns := map[string]bool{}
	add := func(value string) {
		patterns[value] = true
		patterns[strings.ReplaceAll(value, "/", `\/`)] = true
		patterns[url.QueryEscape(value)] = true
	}
	for _, link := range excelImageURLPattern.FindAllString(string(wire), -1) {
		u, err := url.Parse(link)
		if err != nil || !strings.HasSuffix(u.Path, ExcelBPSImagePath) || !validExcelImageToken(u.Query().Get("token")) {
			continue
		}
		add(link)
		add(u.Scheme + "://" + u.Host + u.Path)
		add(u.Path + "?" + u.RawQuery)
		add(u.Query().Get("token"))
	}
	if len(patterns) == 0 {
		return nil
	}
	add(ExcelBPSImagePath)
	r := &excelImageRedactor{streams: make(map[string]*excelImageDelta)}
	for pattern := range patterns {
		r.patterns = append(r.patterns, pattern)
	}
	sort.Slice(r.patterns, func(i, j int) bool { return len(r.patterns[i]) > len(r.patterns[j]) })
	return r
}

// Hold only a possible sensitive prefix, even when the URL/token is split at
// every character boundary. Do not buffer an entire answer or unrelated text.
func (r *excelImageRedactor) replace(text string, final bool) (string, string) {
	var out strings.Builder
	for i := 0; i < len(text); {
		rest := text[i:]
		match := ""
		partial := false
		for _, pattern := range r.patterns {
			if strings.HasPrefix(rest, pattern) && match == "" {
				match = pattern
			}
			if len(rest) < len(pattern) && strings.HasPrefix(pattern, rest) {
				partial = true
			}
		}
		if partial && !final {
			return out.String(), rest
		}
		if match != "" {
			out.WriteString(excelImageReference)
			i += len(match)
			continue
		}
		// Never release almost-complete credentials on a done/failed event. Short
		// prefixes such as the final letter of a normal word are safe to flush.
		if final && partial && len(rest) >= 16 {
			out.WriteString(excelImageReference)
			break
		}
		out.WriteByte(text[i])
		i++
	}
	return out.String(), ""
}

func (r *excelImageRedactor) clean(value any) any {
	switch v := value.(type) {
	case string:
		result, _ := r.replace(v, true)
		return result
	case []any:
		for i, item := range v {
			v[i] = r.clean(item)
		}
		return v
	case map[string]any:
		cleaned := make(map[string]any, len(v))
		for key, item := range v {
			safeKey, _ := r.replace(key, true)
			cleaned[safeKey] = r.clean(item)
		}
		return cleaned
	default:
		return value
	}
}

func (r *excelImageRedactor) payload(raw []byte) (map[string]any, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var payload map[string]any
	if err := decoder.Decode(&payload); err != nil || payload == nil {
		return nil, fmt.Errorf("invalid image response event")
	}
	return payload, nil
}

func (r *excelImageRedactor) redactJSON(raw []byte) []byte {
	if r == nil {
		return raw
	}
	payload, err := r.payload(raw)
	if err != nil {
		return []byte(`{"error":{"message":"Upstream error details unavailable"}}`)
	}
	safe, err := json.Marshal(r.clean(payload))
	if err != nil {
		return []byte(`{}`)
	}
	return safe
}

func (r *excelImageRedactor) events(raw []byte) ([][]byte, error) {
	p, err := r.payload(raw)
	if err != nil {
		return nil, err
	}
	kind, _ := p["type"].(string)
	terminal := kind == "response.completed" || kind == "response.failed" || kind == "response.incomplete" || kind == "error"
	var output [][]byte
	emit := func(p map[string]any) error {
		p = r.clean(p).(map[string]any)
		p["sequence_number"] = r.sequence
		r.sequence++
		encoded, err := json.Marshal(p)
		output = append(output, encoded)
		return err
	}
	if terminal || strings.HasSuffix(kind, ".done") {
		for _, key := range r.order {
			state := r.streams[key]
			if state == nil {
				continue
			}
			if !terminal {
				itemID := p["item_id"]
				if item, ok := p["item"].(map[string]any); ok {
					itemID = item["id"]
				}
				if itemID != nil && !reflect.DeepEqual(itemID, state.template["item_id"]) {
					continue
				}
				if itemID == nil && !reflect.DeepEqual(p["output_index"], state.template["output_index"]) {
					continue
				}
				if kind != "response.output_item.done" {
					if !reflect.DeepEqual(p["content_index"], state.template["content_index"]) || !reflect.DeepEqual(p["summary_index"], state.template["summary_index"]) {
						continue
					}
					if kind != "response.content_part.done" && kind != "response.reasoning_summary_part.done" && strings.TrimSuffix(kind, ".done")+".delta" != state.template["type"] {
						continue
					}
				}
			}
			if state.pending != "" {
				state.template["delta"], _ = r.replace(state.pending, true)
				if err := emit(state.template); err != nil {
					return nil, err
				}
			}
			delete(r.streams, key)
		}
	}
	if delta, ok := p["delta"].(string); ok && strings.HasSuffix(kind, ".delta") {
		keyBytes, _ := json.Marshal([]any{kind, p["item_id"], p["output_index"], p["content_index"], p["summary_index"]})
		key := string(keyBytes)
		state := r.streams[key]
		if state == nil {
			if len(r.order) >= 1024 {
				return nil, fmt.Errorf("too many image response streams")
			}
			state = &excelImageDelta{}
			r.streams[key] = state
			r.order = append(r.order, key)
		}
		state.template = make(map[string]any, len(p))
		for key, value := range p {
			state.template[key] = value
		}
		p["delta"], state.pending = r.replace(state.pending+delta, false)
	}
	if err := emit(p); err != nil {
		return nil, err
	}
	return output, nil
}
