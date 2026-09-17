package service

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/turnstate"
)

func (account *Account) IsCodexTurnStateAutoEnabled() bool {
	if account == nil || !account.IsOpenAIOAuthLike() {
		return false
	}
	enabled, _ := account.Extra[turnstate.EnabledKey].(bool)
	return enabled
}

func (account *Account) InitializeCodexTurnStateAuto() {
	if account == nil || !account.IsOpenAIOAuthLike() {
		return
	}
	if _, exists := account.Extra[turnstate.EnabledKey]; exists {
		return
	}
	extra := make(map[string]any, len(account.Extra)+1)
	for key, value := range account.Extra {
		extra[key] = value
	}
	extra[turnstate.EnabledKey] = true
	account.Extra = extra
}

func (gateway *OpenAIGatewayService) prepareTurnStateSample(ctx context.Context, accountID int64, headers http.Header) (http.Header, bool) {
	if gateway.accountRepo == nil || gateway.openAITokenProvider == nil {
		return nil, false
	}
	account, err := gateway.accountRepo.GetByID(ctx, accountID)
	if err != nil || !account.IsCodexTurnStateAutoEnabled() || account.Status != StatusActive {
		return nil, false
	}
	token := account.GetOpenAIAccessToken()
	if account.Type == AccountTypeOAuth {
		token, err = gateway.openAITokenProvider.GetAccessToken(ctx, account)
	}
	if err != nil || token == "" {
		return nil, false
	}
	headers = headers.Clone()
	headers.Set("Authorization", "Bearer "+token)
	if err := resolveAndSetOpenAIChatGPTAccountHeaders(ctx, gateway.accountRepo, headers, account); err != nil {
		return nil, false
	}
	return headers, true
}

func (gateway *OpenAIGatewayService) applyTurnStateAuto(ctx context.Context, account *Account, model string, headers http.Header) bool {
	if gateway == nil || gateway.turnStateAuto == nil {
		return false
	}
	if !account.IsCodexTurnStateAutoEnabled() {
		if account != nil {
			gateway.turnStateAuto.Forget(account.ID)
		}
		return false
	}
	return gateway.turnStateAuto.Apply(ctx, account.ID, model, headers)
}

func requestTurnStateModel(request *http.Request) string {
	if request == nil || request.GetBody == nil {
		return ""
	}
	body, err := request.GetBody()
	if err != nil {
		return ""
	}
	defer func() { _ = body.Close() }()
	decoder := json.NewDecoder(io.LimitReader(body, 64<<20))
	first, err := decoder.Token()
	if err != nil || first != json.Delim('{') {
		return ""
	}
	for decoder.More() {
		key, err := decoder.Token()
		if err != nil {
			return ""
		}
		if key == "model" {
			var model string
			if decoder.Decode(&model) == nil {
				return strings.TrimSpace(model)
			}
			return ""
		}
		var skipped json.RawMessage
		if decoder.Decode(&skipped) != nil {
			return ""
		}
	}
	return ""
}

func (gateway *OpenAIGatewayService) applyTurnStateAutoRequest(request *http.Request, account *Account) {
	if gateway == nil || gateway.turnStateAuto == nil || request == nil || request.URL == nil {
		return
	}
	if !account.IsCodexTurnStateAutoEnabled() {
		if account != nil {
			gateway.turnStateAuto.Forget(account.ID)
		}
		return
	}
	if request.Method != http.MethodPost || request.URL.Hostname() != "chatgpt.com" || !strings.HasPrefix(request.URL.Path, "/backend-api/codex/responses") {
		return
	}
	gateway.applyTurnStateAuto(request.Context(), account, requestTurnStateModel(request), request.Header)
}

func (gateway *OpenAIGatewayService) TurnStateAutoStatus(accountID int64) map[string]any {
	if gateway == nil || gateway.turnStateAuto == nil {
		return map[string]any{"configured": false, "models": []turnstate.Status{}}
	}
	return map[string]any{"configured": true, "models": gateway.turnStateAuto.Snapshot(accountID)}
}

func (gateway *OpenAIGatewayService) CloseTurnStateAuto() {
	if gateway != nil && gateway.turnStateAuto != nil {
		gateway.turnStateAuto.Close()
	}
}

func reportTurnStateConfigurationError(err error) {
	slog.Error("automatic turn-state disabled due to configuration error", "error", err)
}
