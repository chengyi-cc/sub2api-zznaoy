package handler

import (
	"context"
	"fmt"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/pkg/ip"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// grokViolationFeeRecordedKey guards against charging the same request twice.
const grokViolationFeeRecordedKey = "ops_grok_violation_fee_recorded"

// recordGrokViolationFeeIfMarked 在 gateway forward 返回后检查 Grok 内容安全拒绝标记，
// 命中且后台开启违规扣费时提交一条按次用量行。标记由网关服务层在透传拒绝后设置；
// 拒绝请求不会走正常 RecordUsage，因此不会重复计费。HTTP 每请求至多记录一次；
// WS 传入 wsTurn>0，按 turn 记录（由 clearGrokViolationFeeTurnState 在 turn 收尾复位）。
func (h *OpenAIGatewayHandler) recordGrokViolationFeeIfMarked(c *gin.Context, apiKey *service.APIKey, account *service.Account, subscription *service.UserSubscription, model string, channelFields service.ChannelUsageFields, requestPayloadHash string, wsTurn int) {
	if h == nil || h.gatewayService == nil || !service.HasGrokViolationFee(c) {
		return
	}
	if apiKey == nil || account == nil || account.Platform != service.PlatformGrok {
		return
	}
	if c.GetBool(grokViolationFeeRecordedKey) {
		return
	}
	c.Set(grokViolationFeeRecordedKey, true)

	stream := false
	if v, ok := c.Get(opsStreamKey); ok {
		if b, ok := v.(bool); ok {
			stream = b
		}
	}
	requestID := c.Writer.Header().Get("X-Request-Id")
	if wsTurn > 0 {
		// 同一 WS 连接共用 X-Request-Id，按 turn 区分，避免多次拒绝被计费去重吞掉。
		if requestID == "" {
			requestID = uuid.NewString()
		}
		requestID = fmt.Sprintf("grok-violation:%s:turn-%d", requestID, wsTurn)
		stream = true
	}
	input := service.GrokViolationFeeInput{
		APIKey:             apiKey,
		Account:            account,
		Subscription:       subscription,
		RequestID:          requestID,
		Model:              clientRequestedModel(c, model),
		Stream:             stream,
		OpenAIWSMode:       wsTurn > 0,
		InboundEndpoint:    GetInboundEndpoint(c),
		UpstreamEndpoint:   resolveOpenAIUpstreamEndpoint(c, account, nil),
		UserAgent:          c.GetHeader("User-Agent"),
		IPAddress:          strings.TrimSpace(ip.GetClientIP(c)),
		SessionID:          service.ExtractClientSessionID(c),
		RequestPayloadHash: requestPayloadHash,
		QuotaPlatform:      service.QuotaPlatform(c.Request.Context(), apiKey),
		APIKeyService:      h.apiKeyService,
		ChannelUsageFields: channelFields,
	}
	gwSvc := h.gatewayService
	h.submitMandatoryUsageRecordTask(c.Request.Context(), func(ctx context.Context) {
		gwSvc.RecordGrokViolationFee(ctx, input)
	})
}

// clearGrokViolationFeeTurnState resets the violation mark and recorded guard
// after a WS turn so the next turn is judged on its own.
func clearGrokViolationFeeTurnState(c *gin.Context) {
	if c == nil {
		return
	}
	service.ClearGrokViolationFee(c)
	c.Set(grokViolationFeeRecordedKey, false)
}
