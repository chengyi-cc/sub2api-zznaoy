package service

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"

	"github.com/tidwall/gjson"
)

func (s *OpenAIGatewayService) handleExcelBPSUnauthorized(ctx context.Context, account *Account, status int, headers http.Header, raw []byte, tokens ...string) {
	if status != http.StatusUnauthorized || account == nil {
		return
	}
	if s.excelOAuthReauth != nil && !account.IsExcelOAuth() {
		if len(tokens) > 0 {
			stateCtx, cancel := openAIAccountStateContext(ctx)
			defer cancel()
			var cache OpenAITokenCache
			if s.openAITokenProvider != nil {
				cache = s.openAITokenProvider.tokenCache
			}
			if err := s.excelOAuthReauth.invalidateExcelAccessToken(stateCtx, account.ID, tokens[0], cache); err != nil {
				slog.Warn("excel_credential_rejection_not_saved", "account_id", account.ID)
			}
		}
		return
	}
	if s.rateLimitService == nil {
		return
	}
	fields := map[string]string{"message": "Excel BPS authentication failed"}
	code := extractUpstreamErrorCode(raw)
	if code == "token_invalidated" || code == "token_revoked" {
		fields["code"] = code
	}
	authError := map[string]any{"error": fields}
	if gjson.GetBytes(raw, "detail").String() == "Unauthorized" {
		authError["detail"] = "Unauthorized"
	}
	body, _ := json.Marshal(authError)
	stateCtx, cancel := openAIAccountStateContext(ctx)
	defer cancel()
	s.rateLimitService.HandleUpstreamError(stateCtx, account, status, headers, body)
}
