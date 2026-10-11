package service

import (
	"context"
	"errors"
	"net/http"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/pkg/openai"
	"github.com/gin-gonic/gin"
)

// Only the primary Excel grant is BPS-only. A Codex account with an independent
// Excel grant still supports native Codex requests.
func (a *Account) IsExcelOAuth() bool {
	return a != nil && a.IsOpenAIOAuthLike() && strings.TrimSpace(a.GetCredential("client_id")) == openai.ExcelClientID
}

var errExcelOAuthRouteUnavailable = errors.New("an enabled Excel BPS route is required for Excel OAuth credentials and the requested model; native Codex is unavailable")

const excelOAuthRouteErrorCode = "excel_oauth_route_unavailable"

func (s *OpenAIGatewayService) validateExcelOAuthRoute(ctx context.Context, account *Account, model string) error {
	if account.IsExcelOAuth() && (!account.IsExcelBPSEnabledForModel(model) || !s.excelBPSGloballyEnabled(ctx)) {
		return errExcelOAuthRouteUnavailable
	}
	return nil
}

func writeExcelOAuthRouteError(c *gin.Context) error {
	c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{
		"type": "invalid_request_error", "code": excelOAuthRouteErrorCode,
		"message": errExcelOAuthRouteUnavailable.Error(),
	}})
	return errExcelOAuthRouteUnavailable
}

// The saved switch records intent while an independent Excel grant is prepared.
const ExcelBPSAuthorizationPendingKey = "openai_excel_bps_authorization_pending"

func (a *Account) IsExcelBPSEnabledForModel(model string) bool { return a.UsesExcelBPSForModel(model) }

func QualityBPSEligible(a *Account) bool {
	return a != nil && a.Platform == PlatformOpenAI && a.Type == AccountTypeOAuth && !a.IsShadow() && !a.IsOpenAIAgentIdentity() && !a.IsOpenAIPersonalAccessToken() && !strings.EqualFold(strings.TrimSpace(a.GetCredential("plan_type")), "free")
}

const SettingKeyExcelBPSEnabled = "excel_bps_enabled"

func (s *OpenAIGatewayService) excelBPSGloballyEnabled(ctx context.Context) bool {
	if s == nil || s.settingService == nil {
		return true
	}
	value, err := s.settingService.settingRepo.GetValue(ctx, SettingKeyExcelBPSEnabled)
	return (err == nil && !isFalseSettingValue(value)) || errors.Is(err, ErrSettingNotFound)
}

const ExcelBPSIgnoreEncryptedContentKey = "openai_excel_bps_ignore_encrypted_content"

func (a *Account) IsExcelBPSIgnoreEncryptedContentEnabled() bool {
	return a.IsExcelBPSEnabled() && a.Extra[ExcelBPSIgnoreEncryptedContentKey] == true
}
