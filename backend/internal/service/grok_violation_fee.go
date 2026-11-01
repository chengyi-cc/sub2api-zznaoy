package service

import (
	"context"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/gin-gonic/gin"
)

// Grok 违规扣费：xAI 对生成前被安全检查拦截的请求按次收取 Usage Guidelines
// Violation Fee（见 https://docs.x.ai/developers/pricing）。开启后，Grok 文本请求
// 命中内容安全拒绝（isGrokContentPolicyRejection）时向调用方写一条按次用量行，
// 金额固定，不乘分组/用户倍率，也不按 token 计算。

// GrokViolationFeeAmountDefault 违规扣费默认金额（USD/次）。
const GrokViolationFeeAmountDefault = 0.12

const opsGrokViolationFeeKey = "ops_grok_violation_fee"

// MarkGrokViolationFee 由网关服务层在透传 Grok 内容安全拒绝后设置，
// handler 在 forward 返回后据此写违规扣费用量行。
func MarkGrokViolationFee(c *gin.Context) {
	if c == nil {
		return
	}
	c.Set(opsGrokViolationFeeKey, true)
}

// ClearGrokViolationFee 清除标记（WS 多轮路径在每个 turn 收尾调用，避免带入下一轮）。
func ClearGrokViolationFee(c *gin.Context) {
	if c == nil {
		return
	}
	c.Set(opsGrokViolationFeeKey, false)
}

// HasGrokViolationFee 报告当前请求是否被标记为 Grok 内容安全拒绝。
func HasGrokViolationFee(c *gin.Context) bool {
	return c != nil && c.GetBool(opsGrokViolationFeeKey)
}

func normalizeGrokViolationFeeAmount(v float64) float64 {
	if v < 0 {
		return 0
	}
	return v
}

// GetGrokViolationFeeAmount 返回本次应收取的违规扣费金额；未开启或金额为 0 时返回 0。
// 每次命中直接读存储，保存后对所有实例立即生效（命中频率低，无需本地缓存）。
func (s *SettingService) GetGrokViolationFeeAmount(ctx context.Context) float64 {
	if s == nil || s.settingRepo == nil {
		return 0
	}
	dbCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), gatewayForwardingDBTimeout)
	defer cancel()
	enabled, err := s.settingRepo.GetValue(dbCtx, SettingKeyGrokViolationFeeEnabled)
	if err != nil || enabled != "true" {
		return 0
	}
	raw, err := s.settingRepo.GetValue(dbCtx, SettingKeyGrokViolationFeeAmount)
	if err != nil {
		return GrokViolationFeeAmountDefault
	}
	return parsePositiveFloat(raw, GrokViolationFeeAmountDefault)
}

// grokViolationFeeCost 构造固定金额的按次费用（不乘任何倍率）。
func grokViolationFeeCost(fee float64) *CostBreakdown {
	return &CostBreakdown{
		TotalCost:   fee,
		ActualCost:  fee,
		BillingMode: string(BillingModePerRequest),
	}
}

// GrokViolationFeeInput 是 Grok 违规扣费用量行的入参。
type GrokViolationFeeInput struct {
	APIKey             *APIKey
	Account            *Account
	Subscription       *UserSubscription
	RequestID          string
	Model              string
	Stream             bool
	InboundEndpoint    string
	UpstreamEndpoint   string
	UserAgent          string
	IPAddress          string
	SessionID          string
	RequestPayloadHash string
	QuotaPlatform      string
	APIKeyService      APIKeyQuotaUpdater
	// OpenAIWSMode 标记 WS 多轮连接中的单个 turn：RequestID 须为该 turn 唯一，
	// 计费去重直接使用它，不回退到连接级的 client/local 请求 ID。
	OpenAIWSMode bool
	ChannelUsageFields
}

// RecordGrokViolationFee 在开启违规扣费时为被 xAI 安全检查拒绝的请求写一条
// tokens=0、按次计费的用量行（request_type=cyber），复用 RecordUsage 完成扣费。
func (s *OpenAIGatewayService) RecordGrokViolationFee(ctx context.Context, in GrokViolationFeeInput) {
	if s == nil || in.APIKey == nil || in.APIKey.User == nil || in.Account == nil || in.Account.Platform != PlatformGrok {
		return
	}
	model := strings.TrimSpace(in.Model)
	if model == "" {
		return
	}
	fee := s.settingService.GetGrokViolationFeeAmount(ctx)
	if fee <= 0 {
		return
	}
	result := &OpenAIForwardResult{
		RequestID:    in.RequestID,
		Model:        model,
		Stream:       in.Stream,
		OpenAIWSMode: in.OpenAIWSMode,
		ViolationFee: fee,
	}
	if err := s.RecordUsage(ctx, &OpenAIRecordUsageInput{
		Result:             result,
		APIKey:             in.APIKey,
		User:               in.APIKey.User,
		Account:            in.Account,
		Subscription:       in.Subscription,
		InboundEndpoint:    in.InboundEndpoint,
		UpstreamEndpoint:   in.UpstreamEndpoint,
		UserAgent:          in.UserAgent,
		IPAddress:          in.IPAddress,
		SessionID:          in.SessionID,
		RequestPayloadHash: in.RequestPayloadHash,
		APIKeyService:      in.APIKeyService,
		QuotaPlatform:      in.QuotaPlatform,
		ChannelUsageFields: in.ChannelUsageFields,
		CyberBlocked:       true,
	}); err != nil {
		logger.LegacyPrintf("service.openai_gateway", "grok violation fee record failed: request_id=%s fee=%g err=%v", in.RequestID, fee, err)
	}
}
