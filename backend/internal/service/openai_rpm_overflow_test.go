package service

import (
	"context"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/stretchr/testify/require"
)

type overflowRPMTestCache struct{ openAIRPMTestCache }

func (c *overflowRPMTestCache) IncrementRPM(_ context.Context, id int64) (int, error) {
	if c.err != nil {
		return 0, c.err
	}
	c.counts[id]++
	return c.counts[id], nil
}

func TestOpenAIRPMOverflowOnlyAfterAllCapacityExhausted(t *testing.T) {
	for _, mode := range []string{"legacy", "legacy_batch", "advanced"} {
		t.Run(mode, func(t *testing.T) {
			resetOpenAIAdvancedSchedulerSettingCacheForTest()
			defer resetOpenAIAdvancedSchedulerSettingCacheForTest()
			accounts := []Account{
				{ID: 1, Platform: PlatformOpenAI, Type: AccountTypeOAuth, Status: StatusActive, Schedulable: true, Concurrency: 10, Extra: map[string]any{"base_rpm": 15, "openai_rpm_overflow": true}},
				{ID: 2, Platform: PlatformOpenAI, Type: AccountTypeOAuth, Status: StatusActive, Schedulable: true, Concurrency: 10, Extra: map[string]any{"base_rpm": 15, "openai_rpm_overflow": true}},
			}
			cache := &overflowRPMTestCache{openAIRPMTestCache{counts: map[int64]int{1: 17, 2: 14}}}
			cfg := &config.Config{}
			cfg.Gateway.Scheduling.LoadBatchEnabled = mode == "legacy_batch"
			cfg.Gateway.OpenAIWS.LBTopK = 1
			svc := &OpenAIGatewayService{cfg: cfg, rpmCache: cache, accountRepo: schedulerTestOpenAIAccountRepo{accounts: accounts}, cache: &schedulerTestGatewayCache{sessionBindings: map[string]int64{}}, concurrencyService: NewConcurrencyService(schedulerTestConcurrencyCache{})}
			if mode == "advanced" {
				svc.rateLimitService = newOpenAIAdvancedSchedulerRateLimitService("true")
			}
			choose := func() (*AccountSelectionResult, error) {
				selection, _, err := svc.SelectAccountWithScheduler(context.Background(), nil, "", "", "gpt-5.1", nil, OpenAIUpstreamTransportAny, false)
				if selection != nil && selection.ReleaseFunc != nil {
					selection.ReleaseFunc()
				}
				return selection, err
			}
			selected, err := choose()
			require.NoError(t, err)
			require.EqualValues(t, 2, selected.Account.ID)
			require.False(t, selected.Account.rpmOverflow)
			cache.counts[2] = 19
			selected, err = choose()
			require.NoError(t, err)
			require.EqualValues(t, 1, selected.Account.ID)
			require.True(t, selected.Account.rpmOverflow)
			allowed, state, err := svc.TryAcquireOpenAIOAuthRPM(context.Background(), selected.Account)
			require.NoError(t, err)
			require.True(t, allowed)
			require.Equal(t, 18, state.Current)
			ctx := WithOpenAIRPMReservation(context.Background(), selected.Account, state)
			require.NoError(t, svc.acquireOpenAIRPMForSend(ctx, selected.Account))
			require.Equal(t, 18, cache.counts[1], "do not charge reservation twice")
			require.NoError(t, svc.acquireOpenAIRPMForSend(ctx, selected.Account))
			require.Equal(t, 19, cache.counts[1], "count every retry")
			accounts[0].Extra["openai_rpm_overflow"] = false
			selected, err = choose()
			require.NoError(t, err)
			require.EqualValues(t, 2, selected.Account.ID)
			accounts[1].Extra["openai_rpm_overflow"] = false
			_, err = choose()
			require.ErrorIs(t, err, ErrOpenAIRPMExhausted)
		})
	}
}

func TestExcelActivationDefaultsAndExplicitOverrides(t *testing.T) {
	extra := ExcelBPSActivationExtra(map[string]any{"openai_excel_bps": true}, false, "manual")
	require.Equal(t, 15, extra["base_rpm"])
	require.Equal(t, true, extra["openai_rpm_overflow"])
	require.Equal(t, true, extra["openai_excel_bps_auto_disable_on_403"])
	require.Equal(t, true, extra["openai_excel_bps_cache_creation_as_input"])
	custom := map[string]any{"openai_excel_bps": true, "base_rpm": 0, "openai_excel_bps_auto_disable_on_403": false}
	result := ExcelBPSActivationExtra(custom, false, "manual")
	require.Equal(t, 0, result["base_rpm"])
	require.Equal(t, false, result["openai_excel_bps_auto_disable_on_403"])
	require.Equal(t, custom, ExcelBPSActivationExtra(custom, true, "manual"))
}
