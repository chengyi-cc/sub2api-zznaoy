package service

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"
)

const settingKeyOpenAIAccountTemplate = "openai_account_form_template_v1"

var ErrOpenAIAccountTemplateInvalid = errors.New("invalid OpenAI account template")

// This is an allowlisted form template, never an arbitrary credential/account patch.
type OpenAIAccountTemplate struct {
	Version int                        `json:"version"`
	Fields  map[string]json.RawMessage `json:"fields"`
}

type NamedOpenAIAccountTemplate struct {
	ID      string                     `json:"id"`
	Name    string                     `json:"name"`
	Enabled bool                       `json:"enabled"`
	Fields  map[string]json.RawMessage `json:"fields"`
}

type OpenAIAccountTemplates struct {
	Version   int                          `json:"version"`
	Templates []NamedOpenAIAccountTemplate `json:"templates"`
}

const MaxOpenAIAccountTemplatesSize = 256 * 1024

var openAIAccountTemplateID = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

func (s *SettingService) GetOpenAIAccountTemplate(ctx context.Context) (*OpenAIAccountTemplates, error) {
	raw, err := s.settingRepo.GetValue(ctx, settingKeyOpenAIAccountTemplate)
	if errors.Is(err, ErrSettingNotFound) || (err == nil && raw == "") {
		return &OpenAIAccountTemplates{Version: 2, Templates: []NamedOpenAIAccountTemplate{}}, nil
	}
	if err != nil {
		return nil, err
	}
	var v OpenAIAccountTemplates
	if err = json.Unmarshal([]byte(raw), &v); err != nil {
		return nil, err
	}
	// Read the original single template without modifying storage. The next save
	// atomically replaces it with the collection, including explicit empty lists.
	if v.Version == 1 {
		var legacy OpenAIAccountTemplate
		if err = json.Unmarshal([]byte(raw), &legacy); err != nil {
			return nil, err
		}
		if err = validateOpenAIAccountTemplate(&legacy); err != nil {
			return nil, err
		}
		v = OpenAIAccountTemplates{Version: 2, Templates: []NamedOpenAIAccountTemplate{}}
		if len(legacy.Fields) > 0 {
			v.Templates = append(v.Templates, NamedOpenAIAccountTemplate{ID: "template-1", Name: "模板1", Enabled: true, Fields: legacy.Fields})
		}
	}
	if err = validateOpenAIAccountTemplates(&v); err != nil {
		return nil, err
	}
	return &v, nil
}
func (s *SettingService) SaveOpenAIAccountTemplate(ctx context.Context, v *OpenAIAccountTemplates) error {
	if err := validateOpenAIAccountTemplates(v); err != nil {
		return err
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if len(raw) > MaxOpenAIAccountTemplatesSize {
		return fmt.Errorf("%w: template is too large", ErrOpenAIAccountTemplateInvalid)
	}
	return s.settingRepo.Set(ctx, settingKeyOpenAIAccountTemplate, string(raw))
}

func validateOpenAIAccountTemplates(v *OpenAIAccountTemplates) error {
	if v == nil || v.Version != 2 || v.Templates == nil || len(v.Templates) > 20 {
		return fmt.Errorf("%w: expected version 2 and at most 20 templates", ErrOpenAIAccountTemplateInvalid)
	}
	ids, names := map[string]bool{}, map[string]bool{}
	for i := range v.Templates {
		t := &v.Templates[i]
		t.Name = strings.TrimSpace(t.Name)
		if !openAIAccountTemplateID.MatchString(t.ID) || ids[t.ID] || t.Name == "" || utf8.RuneCountInString(t.Name) > 64 || names[t.Name] || strings.ContainsAny(t.Name, "\r\n\t") {
			return fmt.Errorf("%w: templates need unique IDs and names (1–64 characters)", ErrOpenAIAccountTemplateInvalid)
		}
		ids[t.ID], names[t.Name] = true, true
		item := OpenAIAccountTemplate{Version: 1, Fields: t.Fields}
		if err := validateOpenAIAccountTemplate(&item); err != nil {
			return err
		}
		t.Fields = item.Fields
		raw, err := json.Marshal(t.Fields)
		if err != nil || len(raw) > 65536 {
			return fmt.Errorf("%w: template fields are too large", ErrOpenAIAccountTemplateInvalid)
		}
	}
	return nil
}
func validateOpenAIAccountTemplate(v *OpenAIAccountTemplate) error {
	if v.Version != 1 {
		return fmt.Errorf("%w: unsupported version", ErrOpenAIAccountTemplateInvalid)
	}
	if v.Fields == nil {
		v.Fields = map[string]json.RawMessage{}
	}
	for key, value := range v.Fields {
		if !validOpenAITemplateField(key, value) {
			return fmt.Errorf("%w: %s", ErrOpenAIAccountTemplateInvalid, key)
		}
	}
	return nil
}
func templateDecode(raw json.RawMessage, out any) bool {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	return d.Decode(out) == nil
}
func templateNumber(raw json.RawMessage, min, max float64, integer, nullable bool) bool {
	if string(raw) == "null" {
		return nullable
	}
	var v float64
	return json.Unmarshal(raw, &v) == nil && v >= min && v <= max && !math.IsNaN(v) && !math.IsInf(v, 0) && (!integer || v == math.Trunc(v))
}
func templateEnum(raw json.RawMessage, values ...string) bool {
	var s string
	if json.Unmarshal(raw, &s) != nil {
		return false
	}
	for _, v := range values {
		if s == v {
			return true
		}
	}
	return false
}

type templateMapping struct {
	From string `json:"from"`
	To   string `json:"to"`
}

func validTemplateMappings(m []templateMapping) bool {
	if len(m) > 200 {
		return false
	}
	seen := map[string]bool{}
	for _, v := range m {
		if strings.TrimSpace(v.From) == "" || strings.TrimSpace(v.To) == "" || len(v.From) > 200 || len(v.To) > 200 || seen[v.From] {
			return false
		}
		seen[v.From] = true
	}
	return true
}
func validOpenAITemplateField(key string, raw json.RawMessage) bool {
	raw = bytes.TrimSpace(raw)
	switch key {
	case "autoPauseOnExpired", "openaiPassthroughEnabled", "openaiFlattenNamespacesEnabled", "openAILongContextBillingEnabled", "openAIImagesUrlToB64JsonEnabled":
		var v bool
		return string(raw) != "null" && json.Unmarshal(raw, &v) == nil
	case "concurrency":
		return templateNumber(raw, 1, 100000, true, false)
	case "load_factor":
		return templateNumber(raw, 1, 10000, true, true)
	case "priority":
		return templateNumber(raw, 0, 100000, true, false)
	case "rate_multiplier":
		return templateNumber(raw, 0, 10000, false, false)
	case "proxy_id":
		return templateNumber(raw, 1, 9007199254740991, true, true)
	case "editQuotaLimit", "editQuotaDailyLimit", "editQuotaWeeklyLimit":
		return templateNumber(raw, 0, 1e12, false, true)
	case "group_ids":
		var ids []int64
		if !templateDecode(raw, &ids) || string(raw) == "null" || len(ids) > 500 {
			return false
		}
		for _, id := range ids {
			if id <= 0 || id > 9007199254740991 {
				return false
			}
		}
		return true
	case "codexFingerprintMode":
		return templateEnum(raw, "off", "device", "machine", "session", "full")
	case "openaiResponsesWebSocketV2Mode":
		return templateEnum(raw, "off", "ctx_pool", "passthrough", "http_bridge")
	case "openAICompactMode":
		return templateEnum(raw, "auto", "force_on", "force_off")
	case "openAIResponsesMode":
		return templateEnum(raw, "auto", "force_responses", "force_chat_completions")
	case "tls":
		var v struct {
			Enabled   *bool  `json:"enabled"`
			ProfileID *int64 `json:"profile_id"`
		}
		return templateDecode(raw, &v) && v.Enabled != nil && (v.ProfileID == nil || (*v.ProfileID > 0 && *v.ProfileID <= 9007199254740991))
	case "codexCLI":
		var v struct {
			Enabled        *bool `json:"enabled"`
			AllowAppServer *bool `json:"allow_app_server"`
		}
		return templateDecode(raw, &v) && v.Enabled != nil && v.AllowAppServer != nil
	case "openAICompactModelMappings":
		var v []templateMapping
		return string(raw) != "null" && templateDecode(raw, &v) && validTemplateMappings(v)
	case "modelConfig":
		var v struct {
			Mode          string            `json:"mode"`
			AllowedModels []string          `json:"allowed_models"`
			Mappings      []templateMapping `json:"mappings"`
		}
		if !templateDecode(raw, &v) || (v.Mode != "whitelist" && v.Mode != "mapping") || len(v.AllowedModels) > 300 || !validTemplateMappings(v.Mappings) {
			return false
		}
		for _, model := range v.AllowedModels {
			if strings.TrimSpace(model) == "" || len(model) > 200 {
				return false
			}
		}
		return true
	case "openAIEndpointCapabilities":
		var v []string
		if !templateDecode(raw, &v) || len(v) < 1 || len(v) > 3 {
			return false
		}
		for _, s := range v {
			if s != "chat_completions" && s != "embeddings" && s != "seedance" {
				return false
			}
		}
		return true
	case "poolConfig":
		var v struct {
			Enabled     *bool  `json:"enabled"`
			RetryCount  int    `json:"retry_count"`
			StatusCodes string `json:"status_codes"`
		}
		if !templateDecode(raw, &v) || v.Enabled == nil || v.RetryCount < 1 || v.RetryCount > 10 || len(v.StatusCodes) > 200 {
			return false
		}
		for _, code := range strings.Fields(strings.ReplaceAll(v.StatusCodes, ",", " ")) {
			n, err := strconv.Atoi(code)
			if err != nil || n < 100 || n > 599 {
				return false
			}
		}
		return true
	}
	return false
}
