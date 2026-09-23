//go:build unit

package service

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestOpenAIAccountTemplatePersistsOnlySelectedFields(t *testing.T) {
	repo := newMockSettingRepo()
	svc := &SettingService{settingRepo: repo}
	ctx := context.Background()
	empty, err := svc.GetOpenAIAccountTemplate(ctx)
	require.NoError(t, err)
	require.Equal(t, 1, empty.Version)
	require.Empty(t, empty.Fields)
	v := &OpenAIAccountTemplate{Version: 1, Fields: map[string]json.RawMessage{
		"codexFingerprintMode": json.RawMessage(`"off"`), "concurrency": json.RawMessage(`8`), "openaiPassthroughEnabled": json.RawMessage(`false`), "proxy_id": json.RawMessage(`null`), "group_ids": json.RawMessage(`[]`),
	}}
	require.NoError(t, svc.SaveOpenAIAccountTemplate(ctx, v))
	persisted, err := svc.GetOpenAIAccountTemplate(ctx)
	require.NoError(t, err)
	require.Equal(t, v, persisted)
	delete(v.Fields, "concurrency")
	require.NoError(t, svc.SaveOpenAIAccountTemplate(ctx, v))
	persisted, err = svc.GetOpenAIAccountTemplate(ctx)
	require.NoError(t, err)
	require.NotContains(t, persisted.Fields, "concurrency")
	require.JSONEq(t, `false`, string(persisted.Fields["openaiPassthroughEnabled"]))
	require.JSONEq(t, `null`, string(persisted.Fields["proxy_id"]))
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
			err := svc.SaveOpenAIAccountTemplate(context.Background(), &OpenAIAccountTemplate{Version: 1, Fields: map[string]json.RawMessage{tc.key: json.RawMessage(tc.value)}})
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
	require.NoError(t, svc.SaveOpenAIAccountTemplate(context.Background(), &v))
	loaded, err := svc.GetOpenAIAccountTemplate(context.Background())
	require.NoError(t, err)
	require.Equal(t, &v, loaded)
	repo.getValueErr = errors.New("storage offline")
	_, err = svc.GetOpenAIAccountTemplate(context.Background())
	require.ErrorContains(t, err, "storage offline")
}
