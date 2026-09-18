package service

import (
	"context"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/turnstate"
	"github.com/stretchr/testify/require"
)

type turnStateTriggerAccountRepo struct {
	AccountRepository
	account *Account
}

func (repo *turnStateTriggerAccountRepo) GetByID(context.Context, int64) (*Account, error) {
	return repo.account, nil
}

func TestTriggerTurnStateAcquisitionRejectsInvalidModelBeforeAccountLookup(test *testing.T) {
	gateway := &OpenAIGatewayService{}
	require.ErrorContains(test, gateway.TriggerTurnStateAcquisition(context.Background(), 42, " "), "有效模型")
}

func TestTriggerTurnStateAcquisitionChecksSavedSettingsAndInitialHeaders(test *testing.T) {
	gateway := turnStateSettingsTestGateway(test, &turnStateSettingsTestRepo{})
	account := &Account{ID: 42, Platform: PlatformOpenAI, Type: AccountTypeSetupToken, Status: StatusActive,
		Credentials: map[string]any{"access_token": "test-only-access-token", "chatgpt_account_id": "workspace-a"},
		Extra:       map[string]any{turnstate.EnabledKey: true}}
	gateway.accountRepo = &turnStateTriggerAccountRepo{account: account}
	headers, options, enabled := gateway.prepareTurnStateSample(context.Background(), 42, nil)
	require.True(test, enabled)
	require.Equal(test, "Bearer test-only-access-token", headers.Get("Authorization"))
	require.Equal(test, "workspace-a", headers.Get("Chatgpt-Account-Id"))
	require.Equal(test, turnstate.ProfileTeam, options.Profile)
	require.ErrorContains(test, gateway.TriggerTurnStateAcquisition(context.Background(), 42, "gpt-6-astra"), "尚未配置")
	view, err := gateway.GetTurnStateSettings(context.Background())
	require.NoError(test, err)
	view.PurchasedEnabled = true
	view.ProxyHost, view.ProxyUsername, view.ProxyPassword = "proxy.example:7778", "test_{country}_{session}", "test-password"
	_, err = gateway.SaveTurnStateSettings(context.Background(), view)
	require.NoError(test, err)
	for _, model := range []string{"codex-auto-review", "gpt-5.6-terra", "gpt-5.4"} {
		require.ErrorContains(test, gateway.TriggerTurnStateAcquisition(context.Background(), 42, model), "已关闭请求头采集")
	}
	account.Extra[turnstate.EnabledKey] = false
	require.ErrorContains(test, gateway.TriggerTurnStateAcquisition(context.Background(), 42, "gpt-6-astra"), "先保存账号")
}
