//go:build unit

package service

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestGetGrokViolationFeeAmount(t *testing.T) {
	tests := []struct {
		name   string
		values map[string]string
		want   float64
	}{
		{name: "missing is disabled", values: map[string]string{}, want: 0},
		{name: "disabled ignores amount", values: map[string]string{SettingKeyGrokViolationFeeEnabled: "false", SettingKeyGrokViolationFeeAmount: "0.12"}, want: 0},
		{name: "enabled with amount", values: map[string]string{SettingKeyGrokViolationFeeEnabled: "true", SettingKeyGrokViolationFeeAmount: "0.2"}, want: 0.2},
		{name: "enabled without amount uses default", values: map[string]string{SettingKeyGrokViolationFeeEnabled: "true"}, want: GrokViolationFeeAmountDefault},
		{name: "enabled with malformed amount uses default", values: map[string]string{SettingKeyGrokViolationFeeEnabled: "true", SettingKeyGrokViolationFeeAmount: "abc"}, want: GrokViolationFeeAmountDefault},
		{name: "negative amount clamps to zero", values: map[string]string{SettingKeyGrokViolationFeeEnabled: "true", SettingKeyGrokViolationFeeAmount: "-1"}, want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := NewSettingService(&settingRepoStub{values: tt.values}, nil)
			require.InDelta(t, tt.want, svc.GetGrokViolationFeeAmount(context.Background()), 1e-12)
		})
	}
	var absent *SettingService
	require.Zero(t, absent.GetGrokViolationFeeAmount(context.Background()))
	require.Zero(t, NewSettingService(&settingRepoStub{err: errors.New("unavailable")}, nil).GetGrokViolationFeeAmount(context.Background()))
}

func newGrokViolationFeeServiceForTest(t *testing.T, values map[string]string) (*OpenAIGatewayService, *openAIRecordUsageLogRepoStub, *openAIRecordUsageUserRepoStub) {
	t.Helper()
	usageRepo := &openAIRecordUsageLogRepoStub{inserted: true}
	userRepo := &openAIRecordUsageUserRepoStub{}
	svc := newOpenAIRecordUsageServiceForTest(usageRepo, userRepo, &openAIRecordUsageSubRepoStub{}, nil)
	svc.settingService = NewSettingService(&settingRepoStub{values: values}, nil)
	return svc, usageRepo, userRepo
}

func TestRecordGrokViolationFee_ChargesFixedAmountWithoutMultiplier(t *testing.T) {
	svc, usageRepo, userRepo := newGrokViolationFeeServiceForTest(t, map[string]string{
		SettingKeyGrokViolationFeeEnabled: "true",
		SettingKeyGrokViolationFeeAmount:  "0.12",
	})
	apiKey := &APIKey{ID: 2, User: &User{ID: 1}, GroupID: ptrInt64ForGrokViolationTest(7), Group: &Group{ID: 7, RateMultiplier: 0.16}}

	svc.RecordGrokViolationFee(context.Background(), GrokViolationFeeInput{
		APIKey:    apiKey,
		Account:   &Account{ID: 3, Platform: PlatformGrok},
		RequestID: "rid-grok-violation",
		Model:     "grok-4.7",
	})

	require.Equal(t, 1, usageRepo.calls)
	log := usageRepo.lastLog
	require.NotNil(t, log)
	require.Equal(t, "grok-4.7", log.Model)
	require.Zero(t, log.InputTokens)
	require.Zero(t, log.OutputTokens)
	require.InDelta(t, 0.12, log.TotalCost, 1e-12)
	require.InDelta(t, 0.12, log.ActualCost, 1e-12, "固定金额，不乘分组倍率 0.16")
	require.Equal(t, 1.0, log.RateMultiplier)
	require.NotNil(t, log.BillingMode)
	require.Equal(t, string(BillingModePerRequest), *log.BillingMode)
	require.Equal(t, RequestTypeCyberBlocked, log.RequestType)
	require.Equal(t, 1, userRepo.deductCalls)
	require.InDelta(t, 0.12, userRepo.lastAmount, 1e-12)
}

func TestRecordGrokViolationFee_SkipsWhenDisabledOrNotGrok(t *testing.T) {
	apiKey := &APIKey{ID: 2, User: &User{ID: 1}}

	svc, usageRepo, _ := newGrokViolationFeeServiceForTest(t, map[string]string{SettingKeyGrokViolationFeeEnabled: "false"})
	svc.RecordGrokViolationFee(context.Background(), GrokViolationFeeInput{APIKey: apiKey, Account: &Account{ID: 3, Platform: PlatformGrok}, Model: "grok-4.7"})
	require.Zero(t, usageRepo.calls, "未开启违规扣费时不记不扣")

	svc, usageRepo, _ = newGrokViolationFeeServiceForTest(t, map[string]string{SettingKeyGrokViolationFeeEnabled: "true"})
	svc.RecordGrokViolationFee(context.Background(), GrokViolationFeeInput{APIKey: apiKey, Account: &Account{ID: 3, Platform: PlatformOpenAI}, Model: "gpt-5.1"})
	svc.RecordGrokViolationFee(context.Background(), GrokViolationFeeInput{Account: &Account{ID: 3, Platform: PlatformGrok}, Model: "grok-4.7"})
	svc.RecordGrokViolationFee(context.Background(), GrokViolationFeeInput{APIKey: apiKey, Account: &Account{ID: 3, Platform: PlatformGrok}})
	require.Zero(t, usageRepo.calls, "非 Grok 账号、缺 APIKey 或模型为空时跳过")
}

func TestGrokContentPolicyRejectionMarksViolationFee(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := `{"code":"permission-denied","error":"I'm sorry, I can't help with that request. Failed check: SAFETY_CHECK_TYPE_CSAM"}`
	newResp := func() *http.Response {
		rec := httptest.NewRecorder()
		rec.WriteHeader(http.StatusForbidden)
		rec.Body.WriteString(body)
		return rec.Result()
	}
	account := &Account{ID: 9, Platform: PlatformGrok, Type: AccountTypeOAuth}
	svc := &OpenAIGatewayService{}

	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	_, err := svc.handleErrorResponse(context.Background(), newResp(), c, account, nil, "grok-4.7")
	require.Error(t, err)
	require.True(t, HasGrokViolationFee(c))

	c2, _ := gin.CreateTestContext(httptest.NewRecorder())
	c2.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	_, err = svc.handleCompatErrorResponse(newResp(), c2, account, writeChatCompletionsError, "grok-4.7")
	require.Error(t, err)
	require.True(t, HasGrokViolationFee(c2))

	c3, _ := gin.CreateTestContext(httptest.NewRecorder())
	require.False(t, HasGrokViolationFee(c3))
}

func TestRecordGrokViolationFee_WSTurnsBillSeparately(t *testing.T) {
	svc, usageRepo, userRepo := newGrokViolationFeeServiceForTest(t, map[string]string{
		SettingKeyGrokViolationFeeEnabled: "true",
		SettingKeyGrokViolationFeeAmount:  "0.12",
	})
	apiKey := &APIKey{ID: 2, User: &User{ID: 1}}
	account := &Account{ID: 3, Platform: PlatformGrok}

	var requestIDs []string
	for _, rid := range []string{"grok-violation:conn-1:turn-1", "grok-violation:conn-1:turn-2"} {
		svc.RecordGrokViolationFee(context.Background(), GrokViolationFeeInput{
			APIKey: apiKey, Account: account, RequestID: rid, Model: "grok-4.7", Stream: true, OpenAIWSMode: true,
		})
		require.NotNil(t, usageRepo.lastLog)
		require.True(t, usageRepo.lastLog.OpenAIWSMode)
		requestIDs = append(requestIDs, usageRepo.lastLog.RequestID)
	}
	require.Equal(t, []string{"grok-violation:conn-1:turn-1", "grok-violation:conn-1:turn-2"}, requestIDs, "WS 每个 turn 用独立计费 ID")
	require.Equal(t, 2, userRepo.deductCalls)
}

func TestGrokWSBridgeContentPolicy403MarksViolationFee(t *testing.T) {
	gin.SetMode(gin.TestMode)
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusForbidden,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(strings.NewReader(
			`{"code":"permission-denied","error":"I'm sorry, I can't help with that request. Failed check: SAFETY_CHECK_TYPE_CSAM"}`,
		)),
	}}
	svc := &OpenAIGatewayService{accountRepo: &grokQuotaAccountRepo{}, httpUpstream: upstream}
	account := &Account{ID: 4722, Platform: PlatformGrok, Type: AccountTypeOAuth, Concurrency: 1}
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodGet, "/v1/responses", nil)
	payload := []byte(`{"type":"response.create","model":"grok-4.5","input":"hi"}`)

	_, err := svc.proxyOpenAIWSHTTPBridgeTurn(
		context.Background(), c, account, "access-token", payload, len(payload),
		"grok-4.5", "", "", "", "cache-id", 1,
		func([]byte) error { return nil },
	)

	require.Error(t, err)
	var failoverErr *UpstreamFailoverError
	require.False(t, errors.As(err, &failoverErr), "内容拒绝不换号")
	require.True(t, HasGrokViolationFee(c))
	ClearGrokViolationFee(c)
	require.False(t, HasGrokViolationFee(c))
}

func ptrInt64ForGrokViolationTest(v int64) *int64 { return &v }
