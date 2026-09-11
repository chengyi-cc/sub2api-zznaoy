package service

import (
	"encoding/json"
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

func TestForwardCodeModeInstructions(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, tc := range []struct {
		name, model, mapped, instructions, system, path, userAgent, wantPromptHead string
		wantDeveloper                                                              bool
	}{
		{name: "sol", model: "gpt-5.6-sol", wantDeveloper: true},
		{name: "astra", model: "gpt-6-astra", wantDeveloper: true},
		{name: "mapped to code mode", model: "client-model", mapped: "gpt-5.6-sol", wantDeveloper: true},
		{name: "mapped away from code mode", model: "gpt-5.6-sol", mapped: "gpt-5.4"},
		{name: "bare astra alias resolves to code mode", model: "gpt-6", wantDeveloper: true},
		{name: "bare sol alias resolves to code mode", model: "gpt-5.6", wantDeveloper: true},
		{name: "codex fallback", model: "gpt-5.3-codex", wantPromptHead: "You are a coding agent running in the Codex CLI"},
		{name: "legacy codex alias resolves to fallback", model: "gpt-5-codex", wantPromptHead: "You are a coding agent running in the Codex CLI"},
		{name: "mapped to codex fallback", model: "gpt-5.6-sol", mapped: "gpt-5.3-codex", wantPromptHead: "You are a coding agent running in the Codex CLI"},
		{name: "caller instructions", model: "gpt-5.6-sol", instructions: "Follow the caller's requirements"},
		{name: "caller system", model: "gpt-5.6-sol", system: "Use concise output"},
		{name: "compact", model: "gpt-5.6-sol", path: "/v1/responses/compact"},
		{name: "native client", model: "gpt-5.6-sol", userAgent: "codex_cli_rs/0.153.4"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			account := &Account{ID: 123, Platform: PlatformOpenAI, Type: AccountTypeOAuth,
				Credentials: map[string]any{"access_token": "test-token"}}
			if tc.mapped != "" {
				account.Credentials["model_mapping"] = map[string]any{tc.model: tc.mapped}
			}
			upstream := &httpUpstreamRecorder{err: errors.New("capture only")}
			gateway := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream}
			c, _ := gin.CreateTestContext(httptest.NewRecorder())
			path := tc.path
			if path == "" {
				path = "/v1/responses"
			}
			c.Request = httptest.NewRequest(http.MethodPost, path, nil)
			c.Request.Header.Set("User-Agent", tc.userAgent)
			input := []any{map[string]any{"role": "user", "content": "hello"}}
			if tc.system != "" {
				input = append([]any{map[string]any{"role": "system", "content": tc.system}}, input...)
			}
			request := map[string]any{"model": tc.model, "input": input}
			if tc.instructions != "" {
				request["instructions"] = tc.instructions
			}
			body, err := json.Marshal(request)
			require.NoError(t, err)
			_, err = gateway.Forward(c.Request.Context(), c, account, body)
			require.Error(t, err)
			require.NotNil(t, upstream.lastReq)
			sent := gjson.ParseBytes(upstream.lastBody)
			if tc.mapped != "" {
				require.Equal(t, tc.mapped, sent.Get("model").String())
			}
			if tc.wantDeveloper {
				require.False(t, sent.Get("instructions").Exists(), "default code-mode prompt belongs in input")
				require.Equal(t, "developer", sent.Get("input.0.role").String())
				require.Equal(t, strings.TrimSpace(openai.CodexBaseInstructionsForModel(sent.Get("model").String())), sent.Get("input.0.content").String())
				require.Equal(t, "hello", sent.Get("input.1.content").String())
			} else if tc.instructions != "" {
				require.Equal(t, tc.instructions, sent.Get("instructions").String())
			} else if tc.system != "" {
				require.Contains(t, sent.Get("instructions").String(), tc.system)
			} else {
				require.NotEmpty(t, sent.Get("instructions").String())
				if tc.wantPromptHead != "" {
					require.True(t, strings.HasPrefix(sent.Get("instructions").String(), tc.wantPromptHead))
				}
			}
		})
	}
}

func TestCodeModeRepeatedTransformClearsEmptyInstructions(t *testing.T) {
	for _, empty := range []any{nil, "", "  "} {
		prompt := strings.TrimSpace(openai.CodexBaseInstructionsForModel("gpt-5.6-sol"))
		body := map[string]any{
			"model": "gpt-5.6-sol", "instructions": empty,
			"input": []any{map[string]any{"role": "developer", "content": prompt}, map[string]any{"role": "user", "content": "hello"}},
		}
		result := applyCodexOAuthTransformWithOptions(body, codexOAuthTransformOptions{UseCodeModeInstructions: true})
		require.NoError(t, result.Error)
		require.NotContains(t, body, "instructions")
		require.Len(t, body["input"], 2)
	}
}

func TestConfiguredCodexPromptUsesRoutedModel(t *testing.T) {
	for alias, routed := range map[string]string{
		"gpt-6":       "gpt-6-astra",
		"gpt-5.6":     "gpt-5.6-sol",
		"gpt-5-codex": "gpt-5.3-codex",
	} {
		t.Run(alias, func(t *testing.T) {
			aliasDescriptor := newConfiguredCodexModelDescriptor(alias)
			routedDescriptor := newConfiguredCodexModelDescriptor(routed)
			require.Equal(t, alias, aliasDescriptor.Slug)
			require.Equal(t, routedDescriptor.ModelMessages.InstructionsTemplate, aliasDescriptor.ModelMessages.InstructionsTemplate)
		})
	}
}
