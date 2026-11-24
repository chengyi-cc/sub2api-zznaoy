package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
)

// 本文件是 fork 对上游「上游声明倍率」探测的扩展，尽量不改动 upstream_billing_probe.go：
//
//  1. 上游是 sub2api：倍率探测成功后再查 /v1/usage，把上游余额写进同一份快照。
//  2. 上游是 NewAPI：它没有 /v1/sub2api/billing（404），改用 NewAPI 的令牌只读接口：
//     - 余额：/api/usage/token/（额度按 /api/status 的 quota_per_unit 折算成美元），
//       旧版本回退 /v1/dashboard/billing/subscription + usage；
//     - 倍率：/api/log/token 里最近一条消费日志实际使用的分组倍率（已含分组特殊倍率），
//       令牌尚未产生消费时，若 /api/pricing 只开放了一个分组则以该分组倍率估算。
//     NewAPI 的结果按 sub2api 快照格式写入（无高峰时段），因此展示、调度与倍率同步
//     都无需区分上游类型。
//
// 所有额外请求都是 best-effort：失败只会少一项数据，不会把原本成功的倍率探测变成失败。

const (
	UpstreamBillingKindSub2API = "sub2api"
	UpstreamBillingKindNewAPI  = "newapi"

	upstreamBalanceExtraMaxBodyBytes  = 512 * 1024
	upstreamBalancePricingMaxBytes    = 4 * 1024 * 1024
	upstreamBalanceLogStreamMaxBytes  = 8 * 1024 * 1024
	newAPILogScanLimit                = 50
	newAPIDefaultQuotaPerUnit         = 500000.0
	newAPIDashboardUnlimitedThreshold = 1e8 - 1
)

// EnableBalanceExtension 打开余额与 NewAPI 回退探测。生产装配（ProvideUpstreamBillingProbeService）
// 默认开启；NewUpstreamBillingProbeService 保持上游「一次探测一个请求」的原始语义。
func (s *UpstreamBillingProbeService) EnableBalanceExtension() {
	if s != nil {
		s.balanceExtEnabled = true
	}
}

// UpstreamWalletTokenSource 按上游 host 提供 NewAPI 系统访问令牌，用来读取用户钱包余额
// （/api/user/self）。目前由站点对账模块实现：复用它在「上游访问令牌」里配置的令牌。
type UpstreamWalletTokenSource interface {
	NewAPIAccessToken(ctx context.Context, host string) (string, bool)
}

type upstreamWalletTokenSlot struct {
	p atomic.Pointer[UpstreamWalletTokenSource]
}

// SetWalletTokenSource 在装配期注入；探测协程可能已在运行，故用原子指针。
func (s *UpstreamBillingProbeService) SetWalletTokenSource(src UpstreamWalletTokenSource) {
	if s == nil || src == nil {
		return
	}
	s.walletTokens.p.Store(&src)
}

type newAPIUserSelfResponse struct {
	Success bool `json:"success"`
	Data    *struct {
		Quota *float64 `json:"quota"`
	} `json:"data"`
}

// fetchNewAPIWalletBalance 用站点对账里为该 host 配置的访问令牌查钱包余额。
// 令牌只发往它绑定的 host（target 的 base_url 与 host 必须一致），且请求禁止跟随跳转。
func (s *UpstreamBillingProbeService) fetchNewAPIWalletBalance(ctx context.Context, target upstreamBillingTarget, root string, quotaPerUnit float64) (float64, bool) {
	srcPtr := s.walletTokens.p.Load()
	if srcPtr == nil {
		return 0, false
	}
	host := siteReconCleanHost(root)
	if host == "" {
		return 0, false
	}
	token, ok := (*srcPtr).NewAPIAccessToken(ctx, host)
	if !ok || strings.TrimSpace(token) == "" {
		return 0, false
	}
	walletTarget := target
	walletTarget.apiKey = strings.TrimSpace(token)
	var self newAPIUserSelfResponse
	if err := s.upstreamBillingExtGetJSON(ctx, walletTarget, root+"/api/user/self", true, upstreamBalanceExtraMaxBodyBytes, &self); err != nil ||
		!self.Success || self.Data == nil || self.Data.Quota == nil || !validBillingNumber(*self.Data.Quota) {
		return 0, false
	}
	return *self.Data.Quota / quotaPerUnit, true
}

// upstreamBillingTarget 是一次探测共享的请求上下文（已校验的 base URL、密钥、代理、TLS 画像）。
type upstreamBillingTarget struct {
	account    *Account
	baseURL    string
	apiKey     string
	proxyURL   string
	tlsProfile *tlsfingerprint.Profile
}

func (s *UpstreamBillingProbeService) upstreamBillingExtGet(
	ctx context.Context,
	target upstreamBillingTarget,
	rawURL string,
	withAuth bool,
) (*http.Response, context.CancelFunc, error) {
	if s.accountTestService == nil || s.accountTestService.httpUpstream == nil || target.account == nil {
		return nil, nil, errors.New("transport unavailable")
	}
	reqCtx, cancel := context.WithTimeout(ctx, upstreamBillingProbeRequestTimeout)
	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, rawURL, nil)
	if err != nil {
		cancel()
		return nil, nil, err
	}
	profile := HTTPUpstreamProfileDefault
	if target.account.Platform == PlatformOpenAI {
		profile = HTTPUpstreamProfileOpenAI
	}
	req = req.WithContext(WithHTTPUpstreamRedirectsDisabled(WithHTTPUpstreamProfile(req.Context(), profile)))
	req.Header.Set("Accept", "application/json")
	if withAuth {
		req.Header.Set("Authorization", "Bearer "+target.apiKey)
		target.account.ApplyHeaderOverrides(req.Header)
	}
	resp, err := s.accountTestService.httpUpstream.DoWithTLS(req, target.proxyURL, target.account.ID, target.account.Concurrency, target.tlsProfile)
	if err != nil || resp == nil || resp.Body == nil {
		cancel()
		if err == nil {
			err = errors.New("empty response")
		}
		return nil, nil, err
	}
	return resp, cancel, nil
}

func (s *UpstreamBillingProbeService) upstreamBillingExtGetJSON(
	ctx context.Context,
	target upstreamBillingTarget,
	rawURL string,
	withAuth bool,
	maxBytes int64,
	out any,
) error {
	resp, cancel, err := s.upstreamBillingExtGet(ctx, target, rawURL, withAuth)
	if err != nil {
		return err
	}
	defer cancel()
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxBytes+1))
	if err != nil {
		return err
	}
	if int64(len(body)) > maxBytes {
		return errors.New("response too large")
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("http %d", resp.StatusCode)
	}
	return json.Unmarshal(body, out)
}

// upstreamBillingSiteRoot 去掉 base URL 末尾的 /v1 等版本段，得到 NewAPI 管理接口（/api/...）所在的站点根。
func upstreamBillingSiteRoot(baseURL string) string {
	parsed, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return ""
	}
	path := strings.TrimRight(parsed.Path, "/")
	if idx := strings.LastIndex(path, "/"); idx >= 0 && isOpenAIAPIVersionSegment(path[idx+1:]) {
		path = path[:idx]
	}
	parsed.Path = path
	parsed.RawPath = ""
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return strings.TrimRight(parsed.String(), "/")
}

func validBillingNumber(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

func setUpstreamBalance(data map[string]any, balance float64, unlimited bool, scope string) {
	if unlimited {
		data["upstream_balance_unlimited"] = true
	} else {
		data["upstream_balance"] = math.Round(balance*10000) / 10000
	}
	if scope != "" {
		data["upstream_balance_scope"] = scope
	}
}

// ---------- sub2api ----------

type sub2apiUsageResponse struct {
	Mode      string   `json:"mode"`
	PlanName  string   `json:"planName"`
	Remaining *float64 `json:"remaining"`
	Balance   *float64 `json:"balance"`
}

// attachSub2APIBalance 在 sub2api 倍率探测成功后补查余额（/v1/usage，与倍率同一个 key 鉴权）。
func (s *UpstreamBillingProbeService) attachSub2APIBalance(ctx context.Context, target upstreamBillingTarget, data map[string]any) {
	if !s.balanceExtEnabled {
		return
	}
	data["upstream_kind"] = UpstreamBillingKindSub2API
	var usage sub2apiUsageResponse
	usageURL := buildOpenAIEndpointURL(target.baseURL, "/v1/usage") + "?days=1"
	if err := s.upstreamBillingExtGetJSON(ctx, target, usageURL, true, upstreamBalanceExtraMaxBodyBytes, &usage); err != nil {
		return
	}
	if usage.Remaining == nil || !validBillingNumber(*usage.Remaining) {
		return
	}
	scope := "key_quota"
	switch {
	case usage.Balance != nil:
		scope = "wallet"
	case usage.Mode == "unrestricted":
		scope = "subscription"
	}
	setUpstreamBalance(data, *usage.Remaining, false, scope)
}

// ---------- NewAPI ----------

type newAPIStatusResponse struct {
	Success bool `json:"success"`
	Data    struct {
		QuotaPerUnit     float64 `json:"quota_per_unit"`
		QuotaDisplayType string  `json:"quota_display_type"`
		USDExchangeRate  float64 `json:"usd_exchange_rate"`
	} `json:"data"`
}

type newAPITokenUsageResponse struct {
	Code bool `json:"code"`
	Data *struct {
		TotalAvailable *float64 `json:"total_available"`
		UnlimitedQuota bool     `json:"unlimited_quota"`
		UserQuota      *struct {
			TotalAvailable *float64 `json:"total_available"`
		} `json:"user_quota"`
	} `json:"data"`
}

type newAPIDashboardSubscription struct {
	HardLimitUSD *float64 `json:"hard_limit_usd"`
}

type newAPIDashboardUsage struct {
	TotalUsage *float64 `json:"total_usage"`
}

type newAPIPricingResponse struct {
	Success    bool               `json:"success"`
	GroupRatio map[string]float64 `json:"group_ratio"`
}

type newAPILogEntry struct {
	Type  int    `json:"type"`
	Group string `json:"group"`
	Other string `json:"other"`
}

const newAPILogTypeConsume = 2

// probeNewAPIBilling 尝试按 NewAPI 读取倍率与余额；两项都拿不到时返回 nil（维持 unsupported）。
func (s *UpstreamBillingProbeService) probeNewAPIBilling(ctx context.Context, target upstreamBillingTarget, now time.Time) map[string]any {
	if !s.balanceExtEnabled {
		return nil
	}
	root := upstreamBillingSiteRoot(target.baseURL)
	if root == "" {
		return nil
	}
	var status *newAPIStatusResponse
	var st newAPIStatusResponse
	if err := s.upstreamBillingExtGetJSON(ctx, target, root+"/api/status", false, upstreamBalanceExtraMaxBodyBytes, &st); err == nil && st.Success {
		status = &st
	}

	balance, unlimited, scope, balanceOK := s.fetchNewAPIBalance(ctx, target, root, status)
	rate, group, source, rateOK := s.fetchNewAPIGroupRate(ctx, target, root)
	if !balanceOK && !rateOK {
		return nil
	}
	data := map[string]any{
		"object":         "newapi.token_billing",
		"schema_version": 1,
		"upstream_kind":  UpstreamBillingKindNewAPI,
		"observed_at":    now.UTC().Format(time.RFC3339Nano),
	}
	if rateOK {
		data["billing_scope"] = "token"
		data["group_rate_multiplier"] = rate
		data["resolved_rate_multiplier"] = rate
		data["peak_rate_enabled"] = false
		data["effective_rate_multiplier"] = rate
		data["rate_source"] = source
		if group != "" {
			data["upstream_group"] = group
		}
	}
	if balanceOK {
		setUpstreamBalance(data, balance, unlimited, scope)
	}
	return data
}

func newAPIQuotaPerUnit(status *newAPIStatusResponse) float64 {
	if status != nil && status.Data.QuotaPerUnit > 0 && validBillingNumber(status.Data.QuotaPerUnit) {
		return status.Data.QuotaPerUnit
	}
	return newAPIDefaultQuotaPerUnit
}

func (s *UpstreamBillingProbeService) fetchNewAPIBalance(
	ctx context.Context,
	target upstreamBillingTarget,
	root string,
	status *newAPIStatusResponse,
) (balance float64, unlimited bool, scope string, ok bool) {
	quotaPerUnit := newAPIQuotaPerUnit(status)

	// tokenUnlimited：/api/usage/token/ 已确认令牌是无限额度、但没给出钱包余额。
	// 此时依次尝试：dashboard 接口（站点关闭「额度查询返回令牌额度」DisplayTokenStatEnabled
	// 时按用户钱包统计）→ 站点对账里为该 host 配置的访问令牌查 /api/user/self；
	// 都拿不到钱包余额才退回显示「无限」。
	tokenUnlimited := false
	var usage newAPITokenUsageResponse
	if err := s.upstreamBillingExtGetJSON(ctx, target, root+"/api/usage/token/", true, upstreamBalanceExtraMaxBodyBytes, &usage); err == nil && usage.Code && usage.Data != nil {
		if usage.Data.UnlimitedQuota {
			// 无限额度令牌实际受用户钱包约束；令牌允许查询用户额度时展示钱包余额。
			// 标准 NewAPI 的 /api/usage/token/ 不返回 user_quota，只有部分魔改版本会带。
			if uq := usage.Data.UserQuota; uq != nil && uq.TotalAvailable != nil && validBillingNumber(*uq.TotalAvailable) {
				return *uq.TotalAvailable / quotaPerUnit, false, "user", true
			}
			tokenUnlimited = true
		} else if usage.Data.TotalAvailable != nil && validBillingNumber(*usage.Data.TotalAvailable) {
			return *usage.Data.TotalAvailable / quotaPerUnit, false, "token", true
		}
	}
	unlimitedToken := func() (float64, bool, string, bool) {
		if wallet, ok := s.fetchNewAPIWalletBalance(ctx, target, root, quotaPerUnit); ok {
			return wallet, false, "user", true
		}
		return 0, true, "token", true
	}
	notFound := func() (float64, bool, string, bool) {
		if tokenUnlimited {
			return unlimitedToken()
		}
		return 0, false, "", false
	}

	// OpenAI 兼容的 dashboard 账单接口（旧版 NewAPI / one-api 的唯一途径），金额单位跟随站点展示设置。
	var sub newAPIDashboardSubscription
	var used newAPIDashboardUsage
	if err := s.upstreamBillingExtGetJSON(ctx, target, buildOpenAIEndpointURL(target.baseURL, "/v1/dashboard/billing/subscription"), true, upstreamBalanceExtraMaxBodyBytes, &sub); err != nil ||
		sub.HardLimitUSD == nil || !validBillingNumber(*sub.HardLimitUSD) {
		return notFound()
	}
	if *sub.HardLimitUSD >= newAPIDashboardUnlimitedThreshold {
		return unlimitedToken()
	}
	if err := s.upstreamBillingExtGetJSON(ctx, target, buildOpenAIEndpointURL(target.baseURL, "/v1/dashboard/billing/usage"), true, upstreamBalanceExtraMaxBodyBytes, &used); err != nil ||
		used.TotalUsage == nil || !validBillingNumber(*used.TotalUsage) {
		return notFound()
	}
	// 无限额度令牌在「按令牌统计」模式下 hard_limit 固定为 1e8（上面已拦截），
	// 能走到这里说明 dashboard 统计的是用户钱包。
	scope = "token"
	if tokenUnlimited {
		scope = "user"
	}
	remaining := *sub.HardLimitUSD - *used.TotalUsage/100
	if status != nil {
		switch strings.ToUpper(status.Data.QuotaDisplayType) {
		case "CNY":
			if status.Data.USDExchangeRate > 0 {
				remaining /= status.Data.USDExchangeRate
			}
		case "TOKENS":
			remaining /= quotaPerUnit
		}
	}
	return remaining, false, scope, true
}

func (s *UpstreamBillingProbeService) fetchNewAPIGroupRate(
	ctx context.Context,
	target upstreamBillingTarget,
	root string,
) (rate float64, group string, source string, ok bool) {
	if rate, group, ok := s.fetchNewAPIRecentLogRate(ctx, target, root); ok {
		return rate, group, "recent_log", true
	}
	var pricing newAPIPricingResponse
	if err := s.upstreamBillingExtGetJSON(ctx, target, root+"/api/pricing", false, upstreamBalancePricingMaxBytes, &pricing); err != nil || !pricing.Success {
		return 0, "", "", false
	}
	if len(pricing.GroupRatio) != 1 {
		return 0, "", "", false
	}
	for name, value := range pricing.GroupRatio {
		if value >= 0 && validBillingNumber(value) {
			return value, name, "pricing_single_group", true
		}
	}
	return 0, "", "", false
}

// fetchNewAPIRecentLogRate 流式读取 /api/log/token（按 id 倒序，最多上千条），只解码前若干条，
// 取最近一条消费日志实际扣费使用的分组倍率。
func (s *UpstreamBillingProbeService) fetchNewAPIRecentLogRate(
	ctx context.Context,
	target upstreamBillingTarget,
	root string,
) (float64, string, bool) {
	resp, cancel, err := s.upstreamBillingExtGet(ctx, target, root+"/api/log/token", true)
	if err != nil {
		return 0, "", false
	}
	defer cancel()
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return 0, "", false
	}
	return parseNewAPIRecentLogRate(io.LimitReader(resp.Body, upstreamBalanceLogStreamMaxBytes))
}

func parseNewAPIRecentLogRate(r io.Reader) (float64, string, bool) {
	dec := json.NewDecoder(r)
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		return 0, "", false
	}
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return 0, "", false
		}
		key, _ := keyTok.(string)
		if key != "data" {
			var skip json.RawMessage
			if err := dec.Decode(&skip); err != nil {
				return 0, "", false
			}
			continue
		}
		if tok, err := dec.Token(); err != nil || tok != json.Delim('[') {
			return 0, "", false
		}
		for i := 0; i < newAPILogScanLimit && dec.More(); i++ {
			var entry newAPILogEntry
			if err := dec.Decode(&entry); err != nil {
				return 0, "", false
			}
			if entry.Type != newAPILogTypeConsume || entry.Other == "" {
				continue
			}
			var other struct {
				GroupRatio *float64 `json:"group_ratio"`
			}
			if err := json.Unmarshal([]byte(entry.Other), &other); err != nil || other.GroupRatio == nil {
				continue
			}
			if value := *other.GroupRatio; value >= 0 && validBillingNumber(value) {
				return value, entry.Group, true
			}
		}
		return 0, "", false
	}
	return 0, "", false
}
