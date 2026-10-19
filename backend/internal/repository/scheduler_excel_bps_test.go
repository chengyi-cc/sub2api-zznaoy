package repository

import (
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestExcelBPSSchedulerProjectionRetainsSettings(t *testing.T) {
	for _, models := range [][]string{{"gpt-6-astra", "gpt-6-sol"}, {}} {
		extra := map[string]any{"openai_excel_bps": true, "openai_excel_bps_models": models, "openai_excel_bps_auto_disable_on_403": true, "openai_excel_bps_cache_creation_as_input": true, "private_other": "omit"}
		filtered := filterSchedulerExtra(extra)
		for _, key := range []string{"openai_excel_bps", "openai_excel_bps_models", "openai_excel_bps_auto_disable_on_403", "openai_excel_bps_cache_creation_as_input"} {
			require.Equal(t, extra[key], filtered[key])
		}
		require.NotContains(t, filtered, "private_other")
		account := &service.Account{Platform: service.PlatformOpenAI, Type: service.AccountTypeOAuth, Extra: filtered}
		require.Equal(t, len(models) > 0, account.UsesExcelBPSForModel("gpt-6-sol"))
	}
}
