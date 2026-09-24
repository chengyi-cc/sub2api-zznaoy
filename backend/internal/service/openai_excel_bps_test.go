package service

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/service/basispoints"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func testExcelBPSAccessToken(t *testing.T, accountID string) string {
	t.Helper()
	payload, err := json.Marshal(map[string]any{
		"https://api.openai.com/auth": map[string]string{"chatgpt_account_id": accountID},
	})
	require.NoError(t, err)
	return "header." + base64.RawURLEncoding.EncodeToString(payload) + ".signature"
}

func TestExcelBPSAccountID(t *testing.T) {
	account := &Account{
		Platform:    PlatformOpenAI,
		Type:        AccountTypeOAuth,
		Credentials: map[string]any{"chatgpt_account_id": "stored-account"},
	}
	require.Equal(t, "stored-account", excelBPSAccountID(account, testExcelBPSAccessToken(t, "jwt-account")))

	delete(account.Credentials, "chatgpt_account_id")
	require.Equal(t, "jwt-account", excelBPSAccountID(account, testExcelBPSAccessToken(t, "jwt-account")))
	require.Empty(t, excelBPSAccountID(account, "not-a-jwt"))
	require.Empty(t, excelBPSAccountID(account, testExcelBPSAccessToken(t, "")))
}

func TestNewExcelBPSRequestHeaders(t *testing.T) {
	token := testExcelBPSAccessToken(t, "jwt-account")
	req, err := newExcelBPSRequest(context.Background(), []byte(`{"model":"gpt-5.6-sol"}`), token, "jwt-account")
	require.NoError(t, err)
	require.Equal(t, http.MethodPost, req.Method)
	require.Equal(t, basispoints.ResponsesURL, req.URL.String())
	require.Equal(t, "Bearer "+token, req.Header.Get("authorization"))
	require.Equal(t, "jwt-account", req.Header.Get("chatgpt-account-id"))
	require.Equal(t, "jwt-account", req.Header.Get("x-openai-account-id"))
	require.Equal(t, "chatgpt", req.Header.Get("x-basispoints-auth-mode"))
}

func excelAccount() *Account {
	return &Account{ID: 300, Platform: PlatformOpenAI, Type: AccountTypeOAuth, Status: StatusActive, Schedulable: true, Concurrency: 10,
		Credentials: map[string]any{"access_token": "test-token", "chatgpt_account_id": "test-account"}, Extra: map[string]any{"openai_excel_bps": true, "openai_passthrough": true}}
}
func TestExcelBPSForwardContract(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprint(stream), func(t *testing.T) {
			wire := "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_excel\",\"status\":\"completed\",\"model\":\"gpt-5.6-sol\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"21\"}]}],\"usage\":{\"input_tokens\":10,\"output_tokens\":2}}}\n\n"
			upstream := &httpUpstreamRecorder{resp: &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(wire))}}
			svc := openAIClientToolsTestService(upstream)
			body := []byte(fmt.Sprintf(`{"model":"gpt-5.6-sol","stream":%v,"reasoning":{"effort":"max"},"input":"test","tools":[{"type":"custom","name":"apply_patch"}]}`, stream))
			rec := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(rec)
			c.Request = httptest.NewRequest("POST", "/v1/responses", bytes.NewReader(body))
			c.Request.Header.Set("x-codex-turn-state", "must-not-leak")
			result, err := svc.Forward(context.Background(), c, excelAccount(), body)
			require.NoError(t, err)
			require.NotNil(t, result)
			require.Equal(t, "bps.openai.com", upstream.lastReq.URL.Host)
			require.Equal(t, "/basispoints/api/responses", upstream.lastReq.URL.Path)
			require.Equal(t, "Bearer test-token", upstream.lastReq.Header.Get("Authorization"))
			require.Empty(t, upstream.lastReq.Header.Get("x-codex-turn-state"))
			require.Equal(t, HTTPUpstreamProfileLongStream, HTTPUpstreamProfileFromContext(upstream.lastReq.Context()))
			require.True(t, HTTPUpstreamRedirectsDisabled(upstream.lastReq.Context()))
			require.False(t, gjson.GetBytes(upstream.lastBody, "tools").Exists())
			require.Equal(t, "xhigh", gjson.GetBytes(upstream.lastBody, "reasoning_effort").String())
			require.Equal(t, "xhigh", *result.ReasoningEffort)
			require.Equal(t, 10, result.Usage.InputTokens)
			require.Contains(t, rec.Body.String(), "21")
		})
	}
}
func TestExcelBPSModelDeniedDoesNotFailover(t *testing.T) {
	upstream := &httpUpstreamRecorder{resp: &http.Response{StatusCode: 403, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(`{"error":{"code":"basispoints_model_access_changed","message":"SECRET_UPSTREAM"}}`))}}
	svc := openAIClientToolsTestService(upstream)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest("POST", "/v1/responses", nil)
	_, err := svc.Forward(context.Background(), c, excelAccount(), []byte(`{"model":"gpt-5.6-sol","input":"x"}`))
	require.Error(t, err)
	var failover *UpstreamFailoverError
	require.NotErrorAs(t, err, &failover)
	require.Equal(t, 403, rec.Code)
	require.Contains(t, rec.Body.String(), "basispoints_model_access_changed")
	require.NotContains(t, rec.Body.String(), "SECRET_UPSTREAM")
}
func TestExcelBPSThreadScopeSeparatesParallelChildren(t *testing.T) {
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest("POST", "/v1/responses", nil)
	c.Request.Header.Set("session_id", "shared-root")
	first, _ := resolveOpenAIWSExecutionScope(c, []byte(`{"client_metadata":{"x-codex-turn-metadata":"{\"thread_id\":\"child-A\"}"}}`), 1)
	second, _ := resolveOpenAIWSExecutionScope(c, []byte(`{"client_metadata":{"x-codex-turn-metadata":"{\"thread_id\":\"child-B\"}"}}`), 1)
	require.NotEmpty(t, first)
	require.NotEqual(t, first, second)
}

func TestExcelBPSAccountEligibilityAndNativeRestore(t *testing.T) {
	var missing *Account
	require.False(t, missing.IsExcelBPSEnabled())
	for _, tc := range []struct {
		name   string
		change func(*Account)
	}{
		{"apikey", func(a *Account) { a.Type = AccountTypeAPIKey }},
		{"setup token", func(a *Account) { a.Type = AccountTypeSetupToken }},
		{"other platform", func(a *Account) { a.Platform = PlatformAnthropic }},
		{"shadow", func(a *Account) { id := int64(1); a.ParentAccountID = &id }},
		{"agent identity", func(a *Account) { a.Credentials["auth_mode"] = "agent_identity" }},
		{"native agent identity", func(a *Account) { a.Credentials["auth_mode"] = OpenAIAuthModeAgentIdentity }},
		{"personal token", func(a *Account) { a.Credentials["auth_mode"] = "personalAccessToken" }},
		{"legacy personal token", func(a *Account) { a.Credentials["openai_auth_mode"] = "personal_access_token" }},
		{"string flag", func(a *Account) { a.Extra["openai_excel_bps"] = "true" }},
		{"off by default", func(a *Account) { delete(a.Extra, "openai_excel_bps") }},
	} {
		t.Run(tc.name, func(t *testing.T) { a := excelAccount(); tc.change(a); require.False(t, a.IsExcelBPSEnabled()) })
	}
	a := excelAccount()
	a.Extra["openai_oauth_responses_websockets_v2_mode"] = "ctx_pool"
	a.Extra["openai_oauth_responses_websockets_v2_enabled"] = true
	require.True(t, a.IsExcelBPSEnabled())
	require.False(t, a.IsOpenAIResponsesWebSocketV2Enabled())
	require.Equal(t, OpenAIWSIngressModeOff, a.ResolveOpenAIResponsesWebSocketV2Mode("ctx_pool"))
	require.True(t, a.IsOpenAIWSForceHTTPEnabled())
	delete(a.Extra, "openai_excel_bps")
	require.True(t, a.IsOpenAIResponsesWebSocketV2Enabled())
	require.Equal(t, "ctx_pool", a.ResolveOpenAIResponsesWebSocketV2Mode("ctx_pool"))
	require.False(t, a.IsOpenAIWSForceHTTPEnabled())
}

func excelBPSTestResponse(output string) *http.Response {
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(output))}
}

func TestExcelBPSCompactPreservesMappingAndIntegerHistory(t *testing.T) {
	wire := "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_compact\",\"status\":\"completed\",\"output\":[{\"type\":\"compaction\",\"encrypted_content\":\"test-encrypted\"}]}}\n\n"
	upstream := &httpUpstreamRecorder{resp: excelBPSTestResponse(wire)}
	svc := openAIClientToolsTestService(upstream)
	a := excelAccount()
	a.Credentials["model_mapping"] = map[string]any{"alias": "gpt-6-astra"}
	a.Credentials["compact_model_mapping"] = map[string]any{"alias": "wrong-model"}
	body := []byte(`{"model":"alias","input":[{"type":"function_call","name":"lookup","call_id":"compact-call","arguments":{"value":9007199254740993}},{"type":"function_call_output","call_id":"compact-call","output":"done"}]}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest("POST", "/v1/responses/compact", bytes.NewReader(body))
	result, err := svc.Forward(context.Background(), c, a, body)
	require.NoError(t, err)
	require.Equal(t, "gpt-6-astra", resolveOpenAIAccountUpstreamModelForRequest(a, "alias", true))
	require.Equal(t, "gpt-6-astra", result.UpstreamModel)
	require.Contains(t, string(upstream.lastBody), "9007199254740993")
	input := gjson.GetBytes(upstream.lastBody, "input").Array()
	require.Equal(t, "compaction_trigger", input[len(input)-1].Get("type").String())
	require.Contains(t, rec.Body.String(), "test-encrypted")
}

func TestExcelBPSInvalidRequestsDoNotReachUpstream(t *testing.T) {
	for _, body := range []string{
		`{"model":"gpt-6-astra","input":[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,aA=="}]}]}`,
		`{"model":"gpt-6-astra","input":"test","previous_response_id":"resp_old"}`,
		`{"model":"gpt-6-astra","input":"test","tool_choice":"required"}`,
	} {
		upstream := &httpUpstreamRecorder{}
		svc := openAIClientToolsTestService(upstream)
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest("POST", "/v1/responses", nil)
		_, err := svc.Forward(context.Background(), c, excelAccount(), []byte(body))
		require.Error(t, err)
		require.Equal(t, 400, rec.Code)
		require.Empty(t, upstream.requests)
	}
}

func TestExcelBPSIncompleteStreamDoesNotRecoverScheduling(t *testing.T) {
	for _, stream := range []bool{true, false} {
		upstream := &httpUpstreamRecorder{resp: excelBPSTestResponse("data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n")}
		svc := openAIClientToolsTestService(upstream)
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest("POST", "/v1/responses", nil)
		result, err := svc.Forward(context.Background(), c, excelAccount(), []byte(fmt.Sprintf(`{"model":"gpt-6-astra","stream":%t,"input":"test"}`, stream)))
		require.Error(t, err)
		require.False(t, result.SucceededForScheduling())
		require.Len(t, upstream.requests, 1)
		if !stream {
			require.Equal(t, 502, rec.Code)
		}
	}
}

func TestExcelBPSProbeUsesSelectedRouteAndCandyGrading(t *testing.T) {
	wire := "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_probe\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"CANDY_RESULT=21\"}]}]}}\n\n"
	upstream := &httpUpstreamRecorder{resp: excelBPSTestResponse(wire)}
	s := &AccountTestService{httpUpstream: upstream, openaiGatewayService: openAIClientToolsTestService(upstream)}
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest("POST", "/test", nil)
	c.Set(candyTestContextKey, &candyTestState{started: time.Now()})
	require.NoError(t, s.testOpenAIAccountConnection(c, excelAccount(), "gpt-6-astra", "", AccountTestModeCandy))
	require.Equal(t, "bps.openai.com", upstream.lastReq.URL.Host)
	require.Contains(t, string(upstream.lastBody), candyTestPrompt[:30])
	require.Equal(t, "pass", candyState(c).result.Verdict)
	require.Equal(t, 21, *candyState(c).result.Actual)
	require.Equal(t, "gpt-6-astra", gjson.GetBytes(upstream.lastBody, "model").String())
}

func TestExcelBPSToolReplayIsolatedByAccountKeyAndThread(t *testing.T) {
	native := map[string]any{"type": "custom_tool_call", "id": "ctc_scope_test", "call_id": "call_scope_test", "name": "functions.exec", "input": "return 9007199254740993n", "status": "completed"}
	wire, err := json.Marshal(map[string]any{"type": "response.completed", "response": map[string]any{"id": "resp_scope_test", "status": "completed", "output": []any{native}}})
	require.NoError(t, err)
	upstream := &httpUpstreamRecorder{resp: excelBPSTestResponse("data: " + string(wire) + "\n\n")}
	svc := openAIClientToolsTestService(upstream)
	tools := `"tools":[{"type":"namespace","name":"functions","tools":[{"type":"custom","name":"exec"}]}]`
	request := func(account *Account, key int64, thread, input string) (*httptest.ResponseRecorder, error) {
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest("POST", "/v1/responses", nil)
		c.Request.Header.Set("session_id", "shared-parent")
		c.Request.Header.Set("thread-id", thread)
		c.Set("api_key", &APIKey{ID: key})
		_, err := svc.Forward(context.Background(), c, account, []byte(`{"model":"gpt-6-astra",`+tools+`,"input":`+input+`}`))
		return rec, err
	}
	rec, err := request(excelAccount(), 4001, "child-A", `"hello"`)
	require.NoError(t, err)
	require.Equal(t, "custom_tool_call", gjson.Get(rec.Body.String(), "output.0.type").String())
	require.Equal(t, "exec", gjson.Get(rec.Body.String(), "output.0.name").String())
	require.Equal(t, "functions", gjson.Get(rec.Body.String(), "output.0.namespace").String())
	outputOnly := `[{"type":"custom_tool_call_output","call_id":"call_scope_test","output":"completed"}]`
	otherAccount := excelAccount()
	otherAccount.ID++
	for _, tc := range []struct {
		account *Account
		key     int64
		thread  string
	}{
		{excelAccount(), 4001, "child-B"}, {excelAccount(), 4002, "child-A"}, {otherAccount, 4001, "child-A"},
	} {
		rec, err := request(tc.account, tc.key, tc.thread, outputOnly)
		require.Error(t, err)
		require.Equal(t, 400, rec.Code)
		require.Contains(t, rec.Body.String(), "start a new conversation")
	}
	require.Len(t, upstream.requests, 1)
	upstream.resp = excelBPSTestResponse("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_scope_done\",\"status\":\"completed\",\"output\":[]}}\n\n")
	_, err = request(excelAccount(), 4001, "child-A", outputOnly)
	require.NoError(t, err)
	require.Len(t, upstream.requests, 2)
	items := gjson.GetBytes(upstream.lastBody, "input").Array()
	var replayed bool
	for _, item := range items {
		if item.Get("type").String() == "function_call" {
			replayed = true
			require.Equal(t, "call_scope_test", item.Get("call_id").String())
			require.Equal(t, "run_officejs", item.Get("name").String())
			envelope := gjson.Parse(gjson.Get(item.Get("arguments").String(), "code").String())
			require.Equal(t, "functions.exec", envelope.Get("name").String())
			require.Equal(t, native["input"], envelope.Get("input").String())
		}
	}
	require.True(t, replayed)
}

func TestExcelBPSCancelClosesUpstream(t *testing.T) {
	reader, writer := io.Pipe()
	defer writer.Close()
	upstream := &httpUpstreamRecorder{resp: &http.Response{StatusCode: 200, Header: http.Header{}, Body: reader}}
	svc := openAIClientToolsTestService(upstream)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	ctx, cancel := context.WithCancel(context.Background())
	c.Request = httptest.NewRequest("POST", "/v1/responses", nil).WithContext(ctx)
	defer cancel()
	timer := time.AfterFunc(20*time.Millisecond, cancel)
	defer timer.Stop()
	result, err := svc.Forward(ctx, c, excelAccount(), []byte(`{"model":"gpt-6-astra","stream":true,"input":"test"}`))
	require.ErrorIs(t, err, context.Canceled)
	require.True(t, result.ClientDisconnect)
	require.False(t, result.SucceededForScheduling())
	_, err = writer.Write([]byte("late"))
	require.Error(t, err, "upstream must be closed after downstream cancellation")
}

func TestExcelBPSRequestsWithoutIdentityDoNotShareReplay(t *testing.T) {
	wire := "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_unscoped\",\"status\":\"completed\",\"output\":[{\"id\":\"fc_unscoped\",\"type\":\"function_call\",\"name\":\"lookup\",\"call_id\":\"call_unscoped\",\"arguments\":\"{}\"}]}}\n\n"
	upstream := &httpUpstreamRecorder{resp: excelBPSTestResponse(wire)}
	svc := openAIClientToolsTestService(upstream)
	for index, input := range []string{`"test"`, `[{"type":"function_call_output","call_id":"call_unscoped","output":"done"}]`} {
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest("POST", "/v1/responses", nil)
		_, err := svc.Forward(context.Background(), c, excelAccount(), []byte(`{"model":"gpt-6-astra","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],"input":`+input+`}`))
		if index == 0 {
			require.NoError(t, err)
		} else {
			require.Error(t, err)
			require.Equal(t, 400, rec.Code)
		}
	}
	require.Len(t, upstream.requests, 1)
}

func TestExcelBPSDisabledUsesNativePassthrough(t *testing.T) {
	a := excelAccount()
	a.Extra["openai_excel_bps"] = false
	upstream := &httpUpstreamRecorder{resp: excelBPSTestResponse("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"native\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n")}
	svc := openAIClientToolsTestService(upstream)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest("POST", "/v1/responses", nil)
	_, err := svc.Forward(context.Background(), c, a, []byte(`{"model":"gpt-6-astra","input":"test"}`))
	require.NoError(t, err)
	require.Equal(t, "chatgpt.com", upstream.lastReq.URL.Host)
	require.Equal(t, "/backend-api/codex/responses", upstream.lastReq.URL.Path)
}
