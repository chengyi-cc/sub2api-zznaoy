//go:build unit

package service

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/stretchr/testify/require"
)

func TestGPT61LongContextExemptionSettings(t *testing.T) {
	for _, raw := range []string{"", "false", "true", "invalid", "TRUE"} {
		t.Run(raw, func(t *testing.T) {
			repo := &settingRepoStub{values: map[string]string{SettingKeyOpenAIGPT61SolLongContextExemptionEnabled: raw}}
			svc := NewSettingService(repo, nil)
			require.Equal(t, raw == "true", svc.IsOpenAIGPT61SolLongContextExemptionEnabled(context.Background()))
			settings, err := NewSettingService(&settingGetAllRepoStub{values: repo.values}, &config.Config{}).GetAllSettings(context.Background())
			require.NoError(t, err)
			require.Equal(t, raw == "true", settings.OpenAIGPT61SolLongContextExemptionEnabled)
			repo.values[SettingKeyOpenAIGPT61SolLongContextExemptionEnabled] = "false"
			require.False(t, svc.IsOpenAIGPT61SolLongContextExemptionEnabled(context.Background()), "a saved change must not retain a cached waiver")
		})
	}
	var absent *SettingService
	require.False(t, absent.IsOpenAIGPT61SolLongContextExemptionEnabled(context.Background()))
	require.False(t, NewSettingService(&settingRepoStub{values: map[string]string{}}, nil).IsOpenAIGPT61SolLongContextExemptionEnabled(context.Background()))
	require.False(t, NewSettingService(&settingRepoStub{err: errors.New("unavailable")}, nil).IsOpenAIGPT61SolLongContextExemptionEnabled(context.Background()))
}

func TestGPT61LongContextExemptionSwitchControlsActualBilling(t *testing.T) {
	for _, state := range []string{"missing", "false", "true"} {
		for _, accountType := range []string{AccountTypeOAuth, AccountTypeSetupToken, AccountTypeAPIKey} {
			for _, inputTokens := range []int{272000, 300000} {
				t.Run(fmt.Sprintf("%s/%s/%d", state, accountType, inputTokens), func(t *testing.T) {
					usage := &openAIRecordUsageLogRepoStub{inserted: true}
					users := &openAIRecordUsageUserRepoStub{}
					svc := newOpenAIRecordUsageServiceForTest(usage, users, &openAIRecordUsageSubRepoStub{}, nil)
					if state != "missing" {
						svc.settingService = NewSettingService(&settingRepoStub{values: map[string]string{SettingKeyOpenAIGPT61SolLongContextExemptionEnabled: state}}, nil)
					}
					err := svc.RecordUsage(context.Background(), &OpenAIRecordUsageInput{
						Result: &OpenAIForwardResult{RequestID: t.Name(), Model: "gpt-6.1-sol", Usage: OpenAIUsage{InputTokens: inputTokens, CacheReadInputTokens: 80000, OutputTokens: 2000}},
						APIKey: openAIRecordUsageAPIKeyWithGroup(svc, 1, true), User: &User{ID: 2},
						Account: &Account{ID: 3, Platform: PlatformOpenAI, Type: accountType, Extra: map[string]any{openAILongContextBillingEnabledKey: true}},
					})
					require.NoError(t, err)
					charged := inputTokens > 272000 && (state != "true" || accountType == AccountTypeAPIKey)
					inputFactor, outputFactor := 1.0, 1.0
					if charged {
						inputFactor, outputFactor = 2, 1.5
					}
					expected := (float64(inputTokens-80000)*2e-6+80000*0.1e-6)*inputFactor + 2000*10e-6*outputFactor
					require.Equal(t, charged, usage.lastLog.LongContextBillingApplied)
					require.InDelta(t, expected, usage.lastLog.TotalCost, 1e-12)
					require.InDelta(t, expected*usage.lastLog.RateMultiplier, users.lastAmount, 1e-12)
					require.Equal(t, 1, users.deductCalls)
				})
			}
		}
	}
}
