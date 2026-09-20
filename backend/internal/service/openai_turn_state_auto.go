package service

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	infraerrors "github.com/Wei-Shaw/sub2api/internal/pkg/errors"
	"github.com/Wei-Shaw/sub2api/internal/pkg/openai"
	"github.com/Wei-Shaw/sub2api/internal/turnstate"
)

func (gateway *OpenAIGatewayService) TriggerTurnStateAcquisition(ctx context.Context, accountID int64, model string) error {
	model = strings.TrimSpace(model)
	if accountID <= 0 || model == "" || len(model) > 256 || strings.ContainsAny(model, "\r\n\t") {
		return infraerrors.BadRequest("TURN_STATE_INVALID_MODEL", "请填写有效模型名称和账号")
	}
	if gateway == nil || gateway.turnStateAuto == nil || gateway.accountRepo == nil {
		return infraerrors.BadRequest("TURN_STATE_UNAVAILABLE", "采集服务不可用")
	}
	account, err := gateway.accountRepo.GetByID(ctx, accountID)
	if err != nil {
		return err
	}
	if !account.IsCodexTurnStateAutoEnabled() || account.Status != StatusActive {
		return infraerrors.BadRequest("TURN_STATE_DISABLED", "请先保存账号并开启自动采集，且账号须处于可用状态")
	}
	options := turnstate.OptionsFromExtra(account.Extra)
	if !gateway.turnStateAuto.Configured(options.Source) {
		return infraerrors.BadRequest("TURN_STATE_UNCONFIGURED", "所选采集出口尚未配置，请先保存采集配置")
	}
	if gateway.turnStateAuto.ModelExcluded(model) {
		return infraerrors.BadRequest("TURN_STATE_MODEL_NOT_INCLUDED", "该模型不在采集名单内，请在采集配置中添加并保存后重试")
	}
	headers := make(http.Header)
	headers.Set("User-Agent", codexCLIUserAgent)
	headers.Set("Originator", openai.CodexDefaultOriginator)
	headers.Set("Version", codexCLIVersion)
	headers, options, enabled := gateway.prepareTurnStateSample(ctx, accountID, headers)
	if !enabled {
		return infraerrors.BadRequest("TURN_STATE_AUTH_UNAVAILABLE", "账号授权不可用，无法开始采集")
	}
	if !gateway.turnStateAuto.Force(ctx, accountID, model, headers, options) {
		return infraerrors.BadRequest("TURN_STATE_NOT_QUEUED", "采集任务未能加入队列，请检查缓存服务或稍后重试")
	}
	return nil
}

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
	if gateway.accountRepo == nil {
		return nil, turnstate.Options{}, false
	}
	account, err := gateway.accountRepo.GetByID(ctx, accountID)
	if err != nil || !account.IsCodexTurnStateAutoEnabled() || account.Status != StatusActive {
		return nil, turnstate.Options{}, false
	}
	token := account.GetOpenAIAccessToken()
	if account.Type == AccountTypeOAuth {
		if gateway.openAITokenProvider == nil {
			return nil, turnstate.Options{}, false
		}
		token, err = gateway.openAITokenProvider.GetAccessToken(ctx, account)
	}
	if err != nil || token == "" {
		return nil, turnstate.Options{}, false
	}
	headers = headers.Clone()
	if headers == nil {
		headers = make(http.Header)
	}
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

func (gateway *OpenAIGatewayService) applyTurnStateAutoRequest(request *http.Request, account *Account) error {
	if gateway == nil || gateway.turnStateAuto == nil || request == nil || request.URL == nil {
		return nil
	}
	if !account.IsCodexTurnStateAutoEnabled() {
		if account != nil {
			gateway.turnStateAuto.Forget(account.ID)
		}
		return nil
	}
	if request.Method != http.MethodPost || request.URL.Hostname() != "chatgpt.com" || !strings.HasPrefix(request.URL.Path, "/backend-api/codex/responses") {
		return nil
	}
	model := requestTurnStateModel(request)
	injected := gateway.applyTurnStateAuto(request.Context(), account, model, request.Header)
	return gateway.requireTurnStateAuto(account, model, injected)
}

const OpenAITurnStateUnavailableReason GatewayFailureReason = "turn_state_unavailable"
const OpenAITurnStateUnavailableMessage = "No eligible account has a valid acquired state for this model. Please retry after acquisition completes."

func (failure *UpstreamFailoverError) IsTurnStateUnavailable() bool {
	return failure != nil && failure.Reason == OpenAITurnStateUnavailableReason
}

func isTurnStateUnavailableError(err error) bool {
	var failure *UpstreamFailoverError
	return errors.As(err, &failure) && failure.IsTurnStateUnavailable()
}

func (gateway *OpenAIGatewayService) requireTurnStateAuto(account *Account, model string, injected bool) error {
	if injected || gateway == nil || !account.IsCodexTurnStateAutoEnabled() || !gateway.turnStateAuto.RequiresValidState(model) {
		return nil
	}
	return &UpstreamFailoverError{StatusCode: http.StatusServiceUnavailable, ClientStatusCode: http.StatusServiceUnavailable,
		Reason: OpenAITurnStateUnavailableReason, ClientMessage: OpenAITurnStateUnavailableMessage,
		ResponseHeaders: http.Header{"Retry-After": []string{"5"}}, RequestScopedTransient: true,
		ResponseBody: []byte(`{"error":{"type":"server_error","code":"turn_state_unavailable","message":"No eligible account has a valid acquired state for this model. Please retry after acquisition completes."}}`)}
}

func stopTurnStateFailoverAfterFirstTurn(err error, turn int) error {
	var failure *UpstreamFailoverError
	if turn > 1 && errors.As(err, &failure) && failure.IsTurnStateUnavailable() {
		failure.NextAccountAction = NextAccountStop
	}
	return err
}

func (gateway *OpenAIGatewayService) turnStateResponseObserver(request *http.Request, account *Account) func(*http.Response) {
	if gateway == nil || gateway.turnStateAuto == nil || !account.IsCodexTurnStateAutoEnabled() || request == nil || request.URL == nil || request.Method != http.MethodPost || request.URL.Hostname() != "chatgpt.com" || !strings.HasPrefix(request.URL.Path, "/backend-api/codex/responses") {
		return nil
	}
	model, sent := requestTurnStateModel(request), request.Header.Clone()
	return func(response *http.Response) {
		if response != nil {
			gateway.observeTurnStateAutoResponse(request.Context(), account, model, sent, response.Header, response.StatusCode)
		}
	}
}

func (gateway *OpenAIGatewayService) observeTurnStateAutoResponse(ctx context.Context, account *Account, model string, sent, received http.Header, status int) {
	if gateway == nil || gateway.turnStateAuto == nil || !account.IsCodexTurnStateAutoEnabled() {
		return
	}
	gateway.turnStateAuto.ObserveResponse(ctx, account.ID, model, sent, received, status, turnstate.OptionsFromExtra(account.Extra))
}

func (gateway *OpenAIGatewayService) TurnStateAutoStatus(ctx context.Context, account *Account, includeHistory bool) map[string]any {
	options := turnstate.OptionsFromExtra(account.Extra)
	var manager *turnstate.Runtime
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
	if gateway != nil && gateway.turnStateSettings != nil {
		gateway.turnStateSettings.close()
	}
	if gateway != nil && gateway.turnStateAuto != nil {
		gateway.turnStateAuto.Close()
	}
}

func reportTurnStateConfigurationError(err error) {
	slog.Error("automatic turn-state disabled due to configuration error", "error", err)
}
