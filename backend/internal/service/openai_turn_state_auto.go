package service

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

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
	extra := make(map[string]any, len(account.Extra)+3)
	for key, value := range account.Extra {
		extra[key] = value
	}
	if _, exists := extra[turnstate.EnabledKey]; !exists {
		extra[turnstate.EnabledKey] = true
	}
	if _, exists := extra[turnstate.ProfileKey]; !exists {
		extra[turnstate.ProfileKey] = turnstate.ProfileTeam
	}
	if _, exists := extra[turnstate.SourceKey]; !exists {
		extra[turnstate.SourceKey] = turnstate.SourcePurchased
	}
	account.Extra = extra
}

func (gateway *OpenAIGatewayService) prepareTurnStateSample(ctx context.Context, accountID int64, headers http.Header) (http.Header, turnstate.Options, bool) {
	if gateway.accountRepo == nil || gateway.openAITokenProvider == nil {
		return nil, turnstate.Options{}, false
	}
	account, err := gateway.accountRepo.GetByID(ctx, accountID)
	if err != nil || !account.IsCodexTurnStateAutoEnabled() || account.Status != StatusActive {
		return nil, turnstate.Options{}, false
	}
	token := account.GetOpenAIAccessToken()
	if account.Type == AccountTypeOAuth {
		token, err = gateway.openAITokenProvider.GetAccessToken(ctx, account)
	}
	if err != nil || token == "" {
		return nil, turnstate.Options{}, false
	}
	headers = headers.Clone()
	headers.Set("Authorization", "Bearer "+token)
	if err := resolveAndSetOpenAIChatGPTAccountHeaders(ctx, gateway.accountRepo, headers, account); err != nil {
		return nil, turnstate.Options{}, false
	}
	return headers, turnstate.OptionsFromExtra(account.Extra), true
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
	return gateway.turnStateAuto.Apply(ctx, account.ID, model, headers, turnstate.OptionsFromExtra(account.Extra))
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

func (gateway *OpenAIGatewayService) TurnStateAutoStatus(ctx context.Context, account *Account, includeHistory bool) map[string]any {
	options := turnstate.OptionsFromExtra(account.Extra)
	var manager *turnstate.Manager
	if gateway != nil {
		manager = gateway.turnStateAuto
	}
	queryCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	result := map[string]any{"configured": manager.Configured(options.Source), "enabled": account.IsCodexTurnStateAutoEnabled(), "profile": options.Profile, "source": options.Source, "models": manager.Inspect(queryCtx, account.ID, options), "sources": map[string]bool{turnstate.SourcePurchased: manager.Configured(turnstate.SourcePurchased), turnstate.SourceIPv6: manager.Configured(turnstate.SourceIPv6)}, "countries": manager.Countries(), "server_time": time.Now().UTC()}
	if includeHistory {
		history, err := manager.History(queryCtx, account.ID)
		result["history"] = history
		if err != nil {
			result["history_error"] = err.Error()
		}
	}
	return result
}

func (gateway *OpenAIGatewayService) CloseTurnStateAuto() {
	if gateway != nil && gateway.turnStateAuto != nil {
		gateway.turnStateAuto.Close()
	}
}

func reportTurnStateConfigurationError(err error) {
	slog.Error("automatic turn-state disabled due to configuration error", "error", err)
}
