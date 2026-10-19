//go:build unit

package service

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/openai"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestMergedCatalogRetirementPreservesBPSCapabilities(t *testing.T) {
	const source = `{"models":[{"slug":"public-astra","multi_agent_version":"v2","multi_agent_reasoning_effort":"xhigh","unknown":true},{"slug":"gpt-5.5","multi_agent_version":"v2"}],"metadata":{"kept":true}}`
	for _, pinned := range []bool{false, true} {
		account := excelBPSManifestAccount()
		mapping, ok := account.Credentials["model_mapping"].(map[string]any)
		require.True(t, ok)
		mapping["gpt-5.5"] = "gpt-5.5"
		group := &Group{ID: 11, Platform: PlatformOpenAI}
		group.CodexModelsManifestConfig.Enabled = pinned
		svc := &OpenAIGatewayService{accountRepo: codexModelsVisibilityAccountRepo{byGroup: map[int64][]Account{11: {account}}}}
		original := []byte(source)
		oldETag := codexModelsManifestBodyETag(original)
		manifest := &OpenAIModelsResponse{Body: original, ETag: oldETag}
		require.NoError(t, svc.MergeGroupConfiguredCodexModels(context.Background(), group, manifest, oldETag))
		require.False(t, manifest.NotModified)
		require.False(t, containsManifestModel(manifest.Body, "gpt-5.5"))
		model := gjson.GetBytes(manifest.Body, `models.#(slug=="public-astra")`)
		require.True(t, model.Exists())
		require.Equal(t, "null", model.Get("multi_agent_version").Raw)
		require.Equal(t, "null", model.Get("multi_agent_reasoning_effort").Raw)
		require.True(t, model.Get("unknown").Bool())
		require.True(t, gjson.GetBytes(manifest.Body, "metadata.kept").Bool())
		require.Equal(t, source, string(original))
		require.Equal(t, codexModelsManifestBodyETag(manifest.Body), manifest.ETag)
		next := &OpenAIModelsResponse{Body: []byte(source), ETag: oldETag}
		require.NoError(t, svc.MergeGroupConfiguredCodexModels(context.Background(), group, next, manifest.ETag))
		require.True(t, next.NotModified)
	}
}

func TestMergedForwardCodeModePreservesLiteSearchHistory(t *testing.T) {
	gin.SetMode(gin.TestMode)
	account := &Account{ID: 123, Platform: PlatformOpenAI, Type: AccountTypeOAuth,
		Credentials: map[string]any{"access_token": "test-token"}}
	upstream := &httpUpstreamRecorder{err: errors.New("capture only")}
	gateway := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream}
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	c.Request.Header.Set(responsesLiteHeader, "true")
	body := []byte(`{"model":"gpt-6.1-sol","input":[{"role":"user","content":"summarize"},{"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","query":"example"}}],"tools":[]}`)
	_, err := gateway.Forward(c.Request.Context(), c, account, body)
	require.Error(t, err)
	require.NotNil(t, upstream.lastReq)
	sent := gjson.ParseBytes(upstream.lastBody)
	require.False(t, sent.Get("instructions").Exists())
	require.Equal(t, "developer", sent.Get("input.0.role").String())
	require.Equal(t, strings.TrimSpace(openai.CodexBaseInstructionsForModel("gpt-6.1-sol")), sent.Get("input.0.content").String())
	require.Equal(t, "web_search_call", sent.Get("input.2.type").String())
	require.False(t, gjsonToolsContainWebSearch(sent.Get("tools")))
	require.Equal(t, "web_search", sent.Get(`input.#(type=="additional_tools").tools.0.type`).String())
	require.Equal(t, "none", sent.Get("tool_choice").String())
}
