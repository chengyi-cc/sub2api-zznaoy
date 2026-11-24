//go:build unit

package service

import (
	"context"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/stretchr/testify/require"
)

type billingRouteResponse struct {
	status int
	body   string
}

// billingRouteStub 按 path 返回固定响应，未登记的 path 一律 404。
type billingRouteStub struct {
	mu       sync.Mutex
	routes   map[string]billingRouteResponse
	requests []*http.Request
}

func (u *billingRouteStub) Do(req *http.Request, _ string, _ int64, _ int) (*http.Response, error) {
	u.mu.Lock()
	u.requests = append(u.requests, req)
	r, ok := u.routes[req.URL.Path]
	u.mu.Unlock()
	if !ok {
		r = billingRouteResponse{status: http.StatusNotFound, body: `{"error":"not found"}`}
	}
	return &http.Response{
		StatusCode: r.status,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(r.body)),
	}, nil
}

func (u *billingRouteStub) DoWithTLS(req *http.Request, proxyURL string, accountID int64, c int, _ *tlsfingerprint.Profile) (*http.Response, error) {
	return u.Do(req, proxyURL, accountID, c)
}

func (u *billingRouteStub) find(path string) *http.Request {
	u.mu.Lock()
	defer u.mu.Unlock()
	for _, r := range u.requests {
		if r.URL.Path == path {
			return r
		}
	}
	return nil
}

func newBalanceProbeFixture(t *testing.T, baseURL string, syncRate bool, routes map[string]billingRouteResponse) (*UpstreamBillingProbeService, *Account, *billingRouteStub) {
	t.Helper()
	initialRate := 1.0
	extra := map[string]any{UpstreamBillingProbeEnabledExtraKey: true}
	if syncRate {
		extra[UpstreamBillingRateSyncEnabledExtraKey] = true
	}
	account := &Account{
		ID:             41,
		Platform:       PlatformOpenAI,
		Type:           AccountTypeAPIKey,
		Status:         StatusActive,
		Concurrency:    1,
		Credentials:    map[string]any{"api_key": "sk-relay", "base_url": baseURL},
		Extra:          extra,
		RateMultiplier: &initialRate,
	}
	repo := &upstreamBillingProbeAccountRepo{accounts: map[int64]*Account{account.ID: account}}
	stub := &billingRouteStub{routes: routes}
	svc := newUpstreamBillingProbeTestService(repo, stub, &upstreamBillingProbeSettingRepo{})
	svc.EnableBalanceExtension()
	fixedNow := time.Date(2026, time.November, 19, 2, 0, 0, 0, time.UTC)
	svc.now = func() time.Time { return fixedNow }
	return svc, account, stub
}

const sub2apiBillingBody = `{
	"object":"sub2api.key_billing","schema_version":1,"billing_scope":"token",
	"group_rate_multiplier":0.085,"resolved_rate_multiplier":0.085,"peak_rate_enabled":false,
	"effective_rate_multiplier":0.085,"observed_at":"2026-11-19T01:59:00Z"}`

func TestUpstreamBalance_Sub2APIAttachesWalletBalance(t *testing.T) {
	svc, _, stub := newBalanceProbeFixture(t, "https://relay.example/v1", false, map[string]billingRouteResponse{
		"/v1/sub2api/billing": {200, sub2apiBillingBody},
		"/v1/usage":           {200, `{"mode":"unrestricted","isValid":true,"planName":"钱包余额","remaining":1045.14,"balance":1045.14,"unit":"USD"}`},
	})
	snapshot, err := svc.ProbeAccount(context.Background(), 41)
	require.NoError(t, err)
	require.Equal(t, UpstreamBillingProbeStatusOK, snapshot.Status)
	require.Equal(t, 0.085, snapshot.Data["resolved_rate_multiplier"])
	require.Equal(t, 1045.14, snapshot.Data["upstream_balance"])
	require.Equal(t, "wallet", snapshot.Data["upstream_balance_scope"])
	require.Equal(t, UpstreamBillingKindSub2API, snapshot.Data["upstream_kind"])
	usageReq := stub.find("/v1/usage")
	require.NotNil(t, usageReq)
	require.Equal(t, "Bearer sk-relay", usageReq.Header.Get("Authorization"))
}

func TestUpstreamBalance_Sub2APIBalanceFailureKeepsRate(t *testing.T) {
	svc, _, _ := newBalanceProbeFixture(t, "https://relay.example", false, map[string]billingRouteResponse{
		"/v1/sub2api/billing": {200, sub2apiBillingBody},
		"/v1/usage":           {500, `{"error":"boom"}`},
	})
	snapshot, err := svc.ProbeAccount(context.Background(), 41)
	require.NoError(t, err)
	require.Equal(t, UpstreamBillingProbeStatusOK, snapshot.Status)
	require.Equal(t, 0.085, snapshot.Data["effective_rate_multiplier"])
	require.NotContains(t, snapshot.Data, "upstream_balance")
}

func TestUpstreamBalance_NewAPIUsesRecentLogGroupRatioAndTokenBalance(t *testing.T) {
	svc, account, stub := newBalanceProbeFixture(t, "https://newapi.example/v1", true, map[string]billingRouteResponse{
		"/api/status":       {200, `{"success":true,"data":{"quota_per_unit":500000,"quota_display_type":"USD"}}`},
		"/api/usage/token/": {200, `{"code":true,"message":"ok","data":{"object":"token_usage","total_available":522570000,"unlimited_quota":false}}`},
		"/api/log/token": {200, `{"success":true,"message":"","data":[
			{"id":9,"type":1,"group":"","other":""},
			{"id":8,"type":2,"group":"vip","other":"{\"model_ratio\":2.5,\"group_ratio\":0.085,\"user_group_ratio\":-1}"},
			{"id":7,"type":2,"group":"default","other":"{\"group_ratio\":1}"}
		]}`},
	})
	snapshot, err := svc.ProbeAccount(context.Background(), 41)
	require.NoError(t, err)
	require.Equal(t, UpstreamBillingProbeStatusOK, snapshot.Status)
	require.Equal(t, http.StatusOK, snapshot.HTTPStatus)
	require.Equal(t, UpstreamBillingKindNewAPI, snapshot.Data["upstream_kind"])
	require.Equal(t, 0.085, snapshot.Data["resolved_rate_multiplier"])
	require.Equal(t, "vip", snapshot.Data["upstream_group"])
	require.Equal(t, "recent_log", snapshot.Data["rate_source"])
	require.Equal(t, 1045.14, snapshot.Data["upstream_balance"])

	// 与 sub2api 快照同构：调度读到的倍率与开启同步后的写回都生效。
	rate, ok := upstreamBillingRateAt(snapshot.Data, time.Now())
	require.True(t, ok)
	require.Equal(t, 0.085, rate)
	require.NotNil(t, account.RateMultiplier)
	require.Equal(t, 0.085, *account.RateMultiplier)

	// /api/... 走站点根（去掉 /v1），/api/status 不带密钥。
	statusReq := stub.find("/api/status")
	require.NotNil(t, statusReq)
	require.Equal(t, "https://newapi.example/api/status", statusReq.URL.String())
	require.Empty(t, statusReq.Header.Get("Authorization"))
	require.Equal(t, "Bearer sk-relay", stub.find("/api/log/token").Header.Get("Authorization"))
}

func TestUpstreamBalance_NewAPIPricingSingleGroupAndUnlimitedToken(t *testing.T) {
	svc, _, _ := newBalanceProbeFixture(t, "https://newapi.example", false, map[string]billingRouteResponse{
		"/api/status":       {200, `{"success":true,"data":{"quota_per_unit":500000}}`},
		"/api/usage/token/": {200, `{"code":true,"data":{"total_available":0,"unlimited_quota":true,"user_quota":{"total_available":10000000,"total_used":5}}}`},
		"/api/log/token":    {200, `{"success":true,"data":[]}`},
		"/api/pricing":      {200, `{"success":true,"data":[],"group_ratio":{"default":0.5},"usable_group":{"default":"默认"}}`},
	})
	snapshot, err := svc.ProbeAccount(context.Background(), 41)
	require.NoError(t, err)
	require.Equal(t, UpstreamBillingProbeStatusOK, snapshot.Status)
	require.Equal(t, 0.5, snapshot.Data["resolved_rate_multiplier"])
	require.Equal(t, "pricing_single_group", snapshot.Data["rate_source"])
	require.Equal(t, 20.0, snapshot.Data["upstream_balance"])
	require.Equal(t, "user", snapshot.Data["upstream_balance_scope"])
}

func TestUpstreamBalance_NewAPIAmbiguousPricingLeavesRateUnknown(t *testing.T) {
	svc, account, _ := newBalanceProbeFixture(t, "https://newapi.example", true, map[string]billingRouteResponse{
		"/api/usage/token/": {200, `{"code":true,"data":{"total_available":0,"unlimited_quota":true}}`},
		"/api/pricing":      {200, `{"success":true,"group_ratio":{"default":1,"vip":0.8}}`},
	})
	snapshot, err := svc.ProbeAccount(context.Background(), 41)
	require.NoError(t, err)
	require.Equal(t, UpstreamBillingProbeStatusOK, snapshot.Status)
	require.NotContains(t, snapshot.Data, "resolved_rate_multiplier")
	require.Equal(t, true, snapshot.Data["upstream_balance_unlimited"])
	_, ok := upstreamBillingRateAt(snapshot.Data, time.Now())
	require.False(t, ok)
	require.Equal(t, 1.0, *account.RateMultiplier, "unknown rate must not be synced")
}

// 标准 NewAPI 的 /api/usage/token/ 不带 user_quota；无限额度令牌要继续问 dashboard，
// 站点按用户统计时拿到的就是钱包余额。
func TestUpstreamBalance_NewAPIUnlimitedTokenFallsBackToDashboardWallet(t *testing.T) {
	svc, _, stub := newBalanceProbeFixture(t, "https://newapi.example", false, map[string]billingRouteResponse{
		"/api/status":                        {200, `{"success":true,"data":{"quota_per_unit":500000,"quota_display_type":"USD"}}`},
		"/api/usage/token/":                  {200, `{"code":true,"data":{"total_available":-1,"unlimited_quota":true}}`},
		"/v1/dashboard/billing/subscription": {200, `{"object":"billing_subscription","hard_limit_usd":120}`},
		"/v1/dashboard/billing/usage":        {200, `{"object":"list","total_usage":9025}`},
	})
	snapshot, err := svc.ProbeAccount(context.Background(), 41)
	require.NoError(t, err)
	require.Equal(t, UpstreamBillingProbeStatusOK, snapshot.Status)
	require.NotContains(t, snapshot.Data, "upstream_balance_unlimited")
	require.Equal(t, 29.75, snapshot.Data["upstream_balance"])
	require.Equal(t, "user", snapshot.Data["upstream_balance_scope"])
	require.NotNil(t, stub.find("/v1/dashboard/billing/usage"))
}

// 站点按令牌统计时，无限额度令牌的 dashboard hard_limit 固定为 1e8，仍只能显示「无限」。
func TestUpstreamBalance_NewAPIUnlimitedTokenDashboardAlsoUnlimited(t *testing.T) {
	svc, _, _ := newBalanceProbeFixture(t, "https://newapi.example", false, map[string]billingRouteResponse{
		"/api/usage/token/":                  {200, `{"code":true,"data":{"total_available":-1,"unlimited_quota":true}}`},
		"/v1/dashboard/billing/subscription": {200, `{"object":"billing_subscription","hard_limit_usd":100000000}`},
	})
	snapshot, err := svc.ProbeAccount(context.Background(), 41)
	require.NoError(t, err)
	require.Equal(t, true, snapshot.Data["upstream_balance_unlimited"])
	require.Equal(t, "token", snapshot.Data["upstream_balance_scope"])
}

type walletTokenSourceStub map[string]string

func (m walletTokenSourceStub) NewAPIAccessToken(_ context.Context, host string) (string, bool) {
	tok, ok := m[host]
	return tok, ok
}

// 默认 NewAPI（按令牌统计）下无限额度令牌只能靠站点对账配置的访问令牌查 /api/user/self。
func TestUpstreamBalance_NewAPIUnlimitedTokenUsesSiteReconAccessToken(t *testing.T) {
	svc, _, stub := newBalanceProbeFixture(t, "https://newapi.example:8443/v1", false, map[string]billingRouteResponse{
		"/api/status":                        {200, `{"success":true,"data":{"quota_per_unit":500000}}`},
		"/api/usage/token/":                  {200, `{"code":true,"data":{"total_available":-1,"unlimited_quota":true}}`},
		"/v1/dashboard/billing/subscription": {200, `{"object":"billing_subscription","hard_limit_usd":100000000}`},
		"/api/user/self":                     {200, `{"success":true,"data":{"id":7,"quota":14875000}}`},
	})
	svc.SetWalletTokenSource(walletTokenSourceStub{"newapi.example:8443": "access-token-x"})
	snapshot, err := svc.ProbeAccount(context.Background(), 41)
	require.NoError(t, err)
	require.NotContains(t, snapshot.Data, "upstream_balance_unlimited")
	require.Equal(t, 29.75, snapshot.Data["upstream_balance"])
	require.Equal(t, "user", snapshot.Data["upstream_balance_scope"])
	self := stub.find("/api/user/self")
	require.NotNil(t, self)
	require.Equal(t, "https://newapi.example:8443/api/user/self", self.URL.String())
	require.Equal(t, "Bearer access-token-x", self.Header.Get("Authorization"))
	// 其余请求仍然只用账号自己的 key
	require.Equal(t, "Bearer sk-relay", stub.find("/api/usage/token/").Header.Get("Authorization"))
}

// 访问令牌只发往它绑定的 host；别的 host 的令牌不会被借用，也不会去请求 /api/user/self。
func TestUpstreamBalance_SiteReconAccessTokenIsHostBound(t *testing.T) {
	svc, _, stub := newBalanceProbeFixture(t, "https://newapi.example", false, map[string]billingRouteResponse{
		"/api/usage/token/": {200, `{"code":true,"data":{"total_available":-1,"unlimited_quota":true}}`},
		"/api/user/self":    {200, `{"success":true,"data":{"quota":14875000}}`},
	})
	svc.SetWalletTokenSource(walletTokenSourceStub{"other.example": "access-token-x"})
	snapshot, err := svc.ProbeAccount(context.Background(), 41)
	require.NoError(t, err)
	require.Equal(t, true, snapshot.Data["upstream_balance_unlimited"])
	require.Nil(t, stub.find("/api/user/self"))
}

// 访问令牌失效（/api/user/self 401）时维持「无限」，不把探测变成失败。
func TestUpstreamBalance_SiteReconAccessTokenRejectedKeepsUnlimited(t *testing.T) {
	svc, _, _ := newBalanceProbeFixture(t, "https://newapi.example", false, map[string]billingRouteResponse{
		"/api/usage/token/": {200, `{"code":true,"data":{"total_available":-1,"unlimited_quota":true}}`},
		"/api/user/self":    {401, `{"success":false,"message":"无权进行此操作"}`},
	})
	svc.SetWalletTokenSource(walletTokenSourceStub{"newapi.example": "stale"})
	snapshot, err := svc.ProbeAccount(context.Background(), 41)
	require.NoError(t, err)
	require.Equal(t, UpstreamBillingProbeStatusOK, snapshot.Status)
	require.Equal(t, true, snapshot.Data["upstream_balance_unlimited"])
}

func TestUpstreamBalance_LegacyDashboardFallbackConvertsCNY(t *testing.T) {
	svc, _, _ := newBalanceProbeFixture(t, "https://oldapi.example/v1", false, map[string]billingRouteResponse{
		"/api/status":                        {200, `{"success":true,"data":{"quota_per_unit":500000,"quota_display_type":"CNY","usd_exchange_rate":7.2}}`},
		"/v1/dashboard/billing/subscription": {200, `{"object":"billing_subscription","hard_limit_usd":72}`},
		"/v1/dashboard/billing/usage":        {200, `{"object":"list","total_usage":1440}`},
	})
	snapshot, err := svc.ProbeAccount(context.Background(), 41)
	require.NoError(t, err)
	require.Equal(t, UpstreamBillingProbeStatusOK, snapshot.Status)
	// (72 - 14.40) CNY / 7.2 = 8 USD
	require.Equal(t, 8.0, snapshot.Data["upstream_balance"])
	require.NotContains(t, snapshot.Data, "resolved_rate_multiplier")
}

func TestUpstreamBalance_NothingAnswersStaysUnsupported(t *testing.T) {
	svc, _, _ := newBalanceProbeFixture(t, "https://plain.example", false, map[string]billingRouteResponse{})
	snapshot, err := svc.ProbeAccount(context.Background(), 41)
	require.NoError(t, err)
	require.Equal(t, UpstreamBillingProbeStatusUnsupported, snapshot.Status)
}

func TestUpstreamBalance_DisabledExtensionKeepsUpstreamBehaviour(t *testing.T) {
	svc, _, stub := newBalanceProbeFixture(t, "https://relay.example", false, map[string]billingRouteResponse{
		"/v1/sub2api/billing": {200, sub2apiBillingBody},
		"/v1/usage":           {200, `{"remaining":1,"balance":1}`},
	})
	svc.balanceExtEnabled = false
	snapshot, err := svc.ProbeAccount(context.Background(), 41)
	require.NoError(t, err)
	require.NotContains(t, snapshot.Data, "upstream_balance")
	require.Len(t, stub.requests, 1)
}

func TestParseNewAPIRecentLogRate_StopsAfterScanLimit(t *testing.T) {
	var b strings.Builder
	b.WriteString(`{"success":true,"data":[`)
	for i := 0; i < newAPILogScanLimit; i++ {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(`{"type":1,"other":""}`)
	}
	b.WriteString(`,{"type":2,"group":"late","other":"{\"group_ratio\":0.3}"}]}`)
	_, _, ok := parseNewAPIRecentLogRate(strings.NewReader(b.String()))
	require.False(t, ok)

	rate, group, ok := parseNewAPIRecentLogRate(strings.NewReader(`{"message":"","data":[{"type":2,"group":"g","other":"{\"group_ratio\":0}"}],"success":true}`))
	require.True(t, ok)
	require.Equal(t, 0.0, rate)
	require.Equal(t, "g", group)
}

func TestUpstreamBillingSiteRoot(t *testing.T) {
	require.Equal(t, "https://a.example", upstreamBillingSiteRoot("https://a.example/v1/"))
	require.Equal(t, "https://a.example/relay", upstreamBillingSiteRoot("https://a.example/relay/v1"))
	require.Equal(t, "https://a.example", upstreamBillingSiteRoot("https://a.example"))
	require.Equal(t, "", upstreamBillingSiteRoot("not a url"))
}
