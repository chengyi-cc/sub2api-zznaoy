package service

import (
	"context"
	"errors"
	"github.com/tidwall/gjson"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestExcelBPSModelSelection(t *testing.T) {
	require.Empty(t, (*Account)(nil).ExcelBPSModels())
	require.False(t, (*Account)(nil).UsesExcelBPSForModel("gpt-6-astra"))
	for _, tt := range []struct {
		name     string
		list     any
		explicit bool
		model    string
		want     bool
	}{
		{"astra default", nil, false, "gpt-6-astra", true},
		{"5.6 sol default", nil, false, "gpt-5.6-sol", true},
		{"5.6 terra default", nil, false, "gpt-5.6-terra", true},
		{"6 sol native", nil, false, "gpt-6-sol", false},
		{"6 luna native", nil, false, "gpt-6-luna", false},
		{"no prefix matching", nil, false, "gpt-6-astra-other", false},
		{"custom json list", []any{"gpt-6-sol"}, true, "gpt-6-sol", true},
		{"custom list replaces defaults", []string{"gpt-6-sol"}, true, "gpt-6-astra", false},
		{"explicit empty", []any{}, true, "gpt-6-astra", false},
		{"invalid type", "gpt-6-astra", true, "gpt-6-astra", false},
		{"explicit null", nil, true, "gpt-6-astra", false},
		{"trim and ignore invalid entries", []any{12, "  gpt-6-sol  ", "gpt-6-sol", ""}, true, "gpt-6-sol", true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			a := excelAccount()
			if tt.explicit {
				a.Extra["openai_excel_bps_models"] = tt.list
			}
			require.Equal(t, tt.want, a.UsesExcelBPSForModel(tt.model))
		})
	}
	a := excelAccount()
	a.Credentials["model_mapping"] = map[string]any{"alias": "gpt-6-astra", "gpt-6-astra": "gpt-6-sol"}
	require.True(t, a.UsesExcelBPSForModel("alias"))
	require.False(t, a.UsesExcelBPSForModel("gpt-6-astra"))
	a.Extra["openai_excel_bps"] = false
	require.False(t, a.UsesExcelBPSForModel("alias"))
}

func TestExcelBPSModelScopePreservesNativeTransport(t *testing.T) {
	a := excelAccount()
	a.Extra["openai_oauth_responses_websockets_v2_mode"] = "ctx_pool"
	a.Extra["openai_oauth_responses_websockets_v2_enabled"] = true
	require.Same(t, a, a.forOpenAIModel("gpt-6-astra"))
	native := a.forOpenAIModel("gpt-6-sol")
	require.NotSame(t, a, native)
	require.True(t, a.IsExcelBPSEnabled())
	require.False(t, native.IsExcelBPSEnabled())
	require.True(t, native.IsOpenAIResponsesWebSocketV2Enabled())
	require.Equal(t, "ctx_pool", native.ResolveOpenAIResponsesWebSocketV2Mode("off"))
	require.False(t, native.IsOpenAIWSForceHTTPEnabled())
	require.Equal(t, true, native.Extra["openai_passthrough"])
	native.Extra["openai_passthrough"] = false
	require.Equal(t, true, a.Extra["openai_passthrough"])
	require.Nil(t, (*Account)(nil).forOpenAIModel("gpt-6-sol"))
	svc := openAIClientToolsTestService(nil)
	svc.cfg.Gateway.OpenAIWS.ModeRouterV2Enabled = true
	for _, model := range []string{"gpt-6-sol", "gpt-6-astra"} {
		require.Equal(t, model == "gpt-6-sol", svc.isOpenAIAccountTransportCompatible(a, OpenAIUpstreamTransportResponsesWebsocketV2Ingress, model))
	}
}

func TestExcelBPSModelRoutingActualForward(t *testing.T) {
	for _, tt := range []struct {
		name, model string
		list        any
		explicit    bool
		host, path  string
	}{
		{"selected", "gpt-6-astra", nil, false, "bps.openai.com", "/basispoints/api/responses"},
		{"unselected", "gpt-6-sol", nil, false, "chatgpt.com", "/backend-api/codex/responses"},
		{"opt in sol", "gpt-6-sol", []string{"gpt-6-sol"}, true, "bps.openai.com", "/basispoints/api/responses"},
		{"opt out all", "gpt-6-astra", []string{}, true, "chatgpt.com", "/backend-api/codex/responses"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			a := excelAccount()
			if tt.explicit {
				a.Extra["openai_excel_bps_models"] = tt.list
			}
			upstream := &httpUpstreamRecorder{resp: excelBPSTestResponse("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_test\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n")}
			svc := openAIClientToolsTestService(upstream)
			c, _ := gin.CreateTestContext(httptest.NewRecorder())
			c.Request = httptest.NewRequest("POST", "/v1/responses", nil)
			result, err := svc.Forward(context.Background(), c, a, []byte(`{"model":"`+tt.model+`","input":"test"}`))
			require.NoError(t, err)
			require.NotNil(t, result)
			require.Equal(t, tt.host, upstream.lastReq.URL.Host)
			require.Equal(t, tt.path, upstream.lastReq.URL.Path)
			require.True(t, a.IsExcelBPSEnabled())
			require.Len(t, upstream.requests, 1)
		})
	}
}

func TestExcelBPSWebsocketModelSwitchGuard(t *testing.T) {
	a := excelAccount()
	a.Credentials["model_mapping"] = map[string]any{"alias": "gpt-6-astra"}
	sentinel := errors.New("mapping denied")
	var turns []int
	original := &OpenAIWSIngressHooks{MapRequestModel: func(turn int, model string) (string, error) {
		turns = append(turns, turn)
		if model == "blocked" {
			return "", sentinel
		}
		if model == "group-alias" {
			return "alias", nil
		}
		return model, nil
	}}
	hooks := withExcelBPSModelGuard(a, original)
	model, err := hooks.MapRequestModel(1, "gpt-6-sol")
	require.NoError(t, err)
	require.Equal(t, "gpt-6-sol", model)
	_, err = hooks.MapRequestModel(2, "group-alias")
	require.ErrorContains(t, err, "HTTP")
	_, err = hooks.MapRequestModel(3, "blocked")
	require.ErrorIs(t, err, sentinel)
	require.Equal(t, []int{1, 2, 3}, turns)
	model, err = original.MapRequestModel(4, "gpt-6-astra")
	require.NoError(t, err)
	require.Equal(t, "gpt-6-astra", model)
	_, err = withExcelBPSModelGuard(a, nil).MapRequestModel(1, "gpt-6-astra")
	require.Error(t, err)
	a.Extra["openai_excel_bps"] = false
	require.Same(t, original, withExcelBPSModelGuard(a, original))
}

func TestExcelBPSAccountProbeRespectsModelSelection(t *testing.T) {
	for _, tt := range []struct {
		model    string
		selected bool
		host     string
	}{
		{"gpt-6-astra", true, "bps.openai.com"},
		{"gpt-6-astra", false, "chatgpt.com"},
		{"gpt-6-sol", false, "chatgpt.com"},
		{"gpt-6-sol", true, "bps.openai.com"},
	} {
		t.Run(tt.model+"/"+tt.host, func(t *testing.T) {
			a := excelAccount()
			a.Extra["openai_excel_bps_models"] = []string{}
			if tt.selected {
				a.Extra["openai_excel_bps_models"] = []string{tt.model}
			}
			wire := `data: {"type":"response.output_text.delta","delta":"CANDY_RESULT=21"}

data: {"type":"response.completed","response":{"id":"resp_probe","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"CANDY_RESULT=21"}]}]}}

`
			upstream := &httpUpstreamRecorder{resp: excelBPSTestResponse(wire)}
			gateway := openAIClientToolsTestService(upstream)
			s := &AccountTestService{httpUpstream: upstream, openaiGatewayService: gateway, cfg: gateway.cfg}
			c, _ := gin.CreateTestContext(httptest.NewRecorder())
			c.Request = httptest.NewRequest("POST", "/test", nil)
			c.Set(candyTestContextKey, &candyTestState{started: time.Now()})
			require.NoError(t, s.testOpenAIAccountConnection(c, a, tt.model, "", AccountTestModeCandy))
			require.Equal(t, tt.host, upstream.lastReq.URL.Host)
			require.Equal(t, tt.model, gjson.GetBytes(upstream.lastBody, "model").String())
			require.Contains(t, string(upstream.lastBody), candyTestPrompt[:30])
			require.Equal(t, "pass", candyState(c).result.Verdict)
			require.True(t, a.IsExcelBPSEnabled())
		})
	}
}
