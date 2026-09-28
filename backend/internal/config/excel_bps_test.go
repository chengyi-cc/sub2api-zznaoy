package config

import (
	"github.com/stretchr/testify/require"
	"testing"
)

func TestExcelBPSConfigEnvironmentAndDefaults(t *testing.T) {
	t.Run("diagnostics off by default", func(t *testing.T) {
		resetViperWithJWTSecret(t)
		t.Setenv("GATEWAY_EXCEL_BPS_LOG_REQUEST_TIMING", "")
		cfg, err := Load()
		require.NoError(t, err)
		require.False(t, cfg.Gateway.ExcelBPS.LogRequestTiming)
	})
	for _, tc := range []struct {
		name, value string
		want        int
		invalid     bool
	}{
		{"default", "", 920000, false}, {"zero", "0", 0, false}, {"override", "320000", 320000, false}, {"negative", "-1", 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resetViperWithJWTSecret(t)
			t.Setenv("GATEWAY_EXCEL_BPS_COMPACTION_THRESHOLD_TOKENS", tc.value)
			t.Setenv("GATEWAY_EXCEL_BPS_LOG_REQUEST_TIMING", "true")
			cfg, err := Load()
			if tc.invalid {
				require.ErrorContains(t, err, "compaction_threshold_tokens")
				return
			}
			require.NoError(t, err)
			require.NotNil(t, cfg.Gateway.ExcelBPS.CompactionThresholdTokens)
			require.Equal(t, tc.want, *cfg.Gateway.ExcelBPS.CompactionThresholdTokens)
			require.True(t, cfg.Gateway.ExcelBPS.LogRequestTiming)
		})
	}
}
