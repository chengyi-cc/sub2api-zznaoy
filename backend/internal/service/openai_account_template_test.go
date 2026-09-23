//go:build unit

package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func templateCollection(fields map[string]json.RawMessage) *OpenAIAccountTemplates {
	return &OpenAIAccountTemplates{Version: 2, Templates: []NamedOpenAIAccountTemplate{{ID: "template-1", Name: "Template 1", Enabled: true, Fields: fields}}}
}

func TestOpenAIAccountTemplatePersistsOnlySelectedFields(t *testing.T) {
	repo := newMockSettingRepo()
	svc := &SettingService{settingRepo: repo}
	ctx := context.Background()
	empty, err := svc.GetOpenAIAccountTemplate(ctx)
	require.NoError(t, err)
	require.Equal(t, 2, empty.Version)
	require.Empty(t, empty.Templates)
	v := templateCollection(map[string]json.RawMessage{
		"codexFingerprintMode": json.RawMessage(`"off"`), "concurrency": json.RawMessage(`8`), "openaiPassthroughEnabled": json.RawMessage(`false`), "proxy_id": json.RawMessage(`null`), "group_ids": json.RawMessage(`[]`),
	})
	require.NoError(t, svc.SaveOpenAIAccountTemplate(ctx, v))
	persisted, err := svc.GetOpenAIAccountTemplate(ctx)
	require.NoError(t, err)
	require.Equal(t, v, persisted)
	delete(v.Templates[0].Fields, "concurrency")
	require.NoError(t, svc.SaveOpenAIAccountTemplate(ctx, v))
	persisted, err = svc.GetOpenAIAccountTemplate(ctx)
	require.NoError(t, err)
	require.NotContains(t, persisted.Templates[0].Fields, "concurrency")
	require.JSONEq(t, `false`, string(persisted.Templates[0].Fields["openaiPassthroughEnabled"]))
	require.JSONEq(t, `null`, string(persisted.Templates[0].Fields["proxy_id"]))
}
func TestOpenAIAccountTemplateRejectsUnknownSecretsAndInvalidValues(t *testing.T) {
	for _, tc := range []struct{ key, value string }{
		{"api_key", `"secret"`}, {"credentials", `{"access_token":"secret"}`}, {"name", `"overwrite"`}, {"status", `"inactive"`},
		{"codexFingerprintMode", `"invalid"`}, {"concurrency", `0`}, {"concurrency", `null`}, {"concurrency", `1.5`}, {"load_factor", `0.5`}, {"openaiPassthroughEnabled", `null`}, {"priority", `null`},
		{"group_ids", `[-1]`}, {"group_ids", `null`}, {"openAIEndpointCapabilities", `[]`}, {"openAIEndpointCapabilities", `["invalid"]`},
		{"tls", `{"enabled":true,"access_token":"secret"}`}, {"tls", `{}`}, {"codexCLI", `{"enabled":true}`},
		{"openAICompactModelMappings", `[{"from":"","to":"model"}]`}, {"modelConfig", `{"mode":"unknown"}`},
	} {
		t.Run(tc.key+tc.value, func(t *testing.T) {
			repo := newMockSettingRepo()
			svc := &SettingService{settingRepo: repo}
			err := svc.SaveOpenAIAccountTemplate(context.Background(), templateCollection(map[string]json.RawMessage{tc.key: json.RawMessage(tc.value)}))
			require.ErrorIs(t, err, ErrOpenAIAccountTemplateInvalid)
			require.Empty(t, repo.data)
		})
	}
}
func TestOpenAIAccountTemplateTransportAndComplexFields(t *testing.T) {
	repo := newMockSettingRepo()
	svc := &SettingService{settingRepo: repo}
	var v OpenAIAccountTemplate
	require.NoError(t, json.Unmarshal([]byte(`{"version":1,"fields":{"tls":{"enabled":true,"profile_id":2},"codexCLI":{"enabled":true,"allow_app_server":false},"modelConfig":{"mode":"mapping","allowed_models":[],"mappings":[{"from":"public","to":"upstream"}]},"openAICompactMode":"force_on","openAIResponsesMode":"force_chat_completions","openAIEndpointCapabilities":["embeddings"],"poolConfig":{"enabled":true,"retry_count":3,"status_codes":"401, 429"}}}`), &v))
	require.NoError(t, svc.SaveOpenAIAccountTemplate(context.Background(), templateCollection(v.Fields)))
	loaded, err := svc.GetOpenAIAccountTemplate(context.Background())
	require.NoError(t, err)
	require.Equal(t, v.Fields, loaded.Templates[0].Fields)
	repo.getValueErr = errors.New("storage offline")
	_, err = svc.GetOpenAIAccountTemplate(context.Background())
	require.ErrorContains(t, err, "storage offline")
}

func TestOpenAIAccountTemplatesMigrateLegacyWithoutLosingFields(t *testing.T) {
	repo := newMockSettingRepo()
	svc := &SettingService{settingRepo: repo}
	ctx := context.Background()
	legacy := `{"version":1,"fields":{"concurrency":3,"openaiPassthroughEnabled":false,"proxy_id":null,"group_ids":[]}}`
	require.NoError(t, repo.Set(ctx, settingKeyOpenAIAccountTemplate, legacy))
	loaded, err := svc.GetOpenAIAccountTemplate(ctx)
	require.NoError(t, err)
	require.Equal(t, 2, loaded.Version)
	require.Len(t, loaded.Templates, 1)
	require.Equal(t, "模板1", loaded.Templates[0].Name)
	require.True(t, loaded.Templates[0].Enabled)
	require.Len(t, loaded.Templates[0].Fields, 4)
	raw, err := repo.GetValue(ctx, settingKeyOpenAIAccountTemplate)
	require.NoError(t, err)
	require.Equal(t, legacy, raw, "reading should not mutate persisted settings")
	loaded.Templates[0].Enabled = false
	loaded.Templates = append(loaded.Templates, NamedOpenAIAccountTemplate{ID: "second", Name: "Second", Enabled: true, Fields: map[string]json.RawMessage{"concurrency": json.RawMessage(`9`)}})
	require.NoError(t, svc.SaveOpenAIAccountTemplate(ctx, loaded))
	persisted, err := svc.GetOpenAIAccountTemplate(ctx)
	require.NoError(t, err)
	require.Equal(t, loaded, persisted)
	loaded.Templates = []NamedOpenAIAccountTemplate{}
	require.NoError(t, svc.SaveOpenAIAccountTemplate(ctx, loaded))
	persisted, err = svc.GetOpenAIAccountTemplate(ctx)
	require.NoError(t, err)
	require.Empty(t, persisted.Templates, "deleting all templates must not restore the legacy template")
	require.NoError(t, repo.Set(ctx, settingKeyOpenAIAccountTemplate, `{"version":1,"fields":{}}`))
	persisted, err = svc.GetOpenAIAccountTemplate(ctx)
	require.NoError(t, err)
	require.Empty(t, persisted.Templates)
}

func TestOpenAIAccountTemplatesRejectInvalidCollections(t *testing.T) {
	for name, mutate := range map[string]func(*OpenAIAccountTemplates){
		"version":        func(v *OpenAIAccountTemplates) { v.Version = 1 },
		"null list":      func(v *OpenAIAccountTemplates) { v.Templates = nil },
		"empty ID":       func(v *OpenAIAccountTemplates) { v.Templates[0].ID = "" },
		"invalid ID":     func(v *OpenAIAccountTemplates) { v.Templates[0].ID = "a/b" },
		"empty name":     func(v *OpenAIAccountTemplates) { v.Templates[0].Name = "  " },
		"long name":      func(v *OpenAIAccountTemplates) { v.Templates[0].Name = strings.Repeat("模", 65) },
		"multiline name": func(v *OpenAIAccountTemplates) { v.Templates[0].Name = "a\nb" },
		"duplicate ID": func(v *OpenAIAccountTemplates) {
			second := v.Templates[0]
			second.Name = "Other"
			v.Templates = append(v.Templates, second)
		},
		"duplicate name": func(v *OpenAIAccountTemplates) {
			second := v.Templates[0]
			second.ID = "other"
			second.Name += " "
			v.Templates = append(v.Templates, second)
		},
		"too many": func(v *OpenAIAccountTemplates) {
			for i := 1; i <= 20; i++ {
				v.Templates = append(v.Templates, NamedOpenAIAccountTemplate{ID: fmt.Sprint(i), Name: fmt.Sprint(i)})
			}
		},
		"disabled invalid fields": func(v *OpenAIAccountTemplates) {
			v.Templates[0].Enabled = false
			v.Templates[0].Fields["access_token"] = json.RawMessage(`"secret"`)
		},
	} {
		t.Run(name, func(t *testing.T) {
			repo := newMockSettingRepo()
			svc := &SettingService{settingRepo: repo}
			v := templateCollection(map[string]json.RawMessage{})
			mutate(v)
			require.ErrorIs(t, svc.SaveOpenAIAccountTemplate(context.Background(), v), ErrOpenAIAccountTemplateInvalid)
			require.Empty(t, repo.data)
		})
	}
}
