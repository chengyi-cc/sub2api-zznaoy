package service

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/turnstate"
	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
)

type turnStateSettingsTestRepo struct {
	SettingRepository
	mu    sync.Mutex
	value string
	fail  bool
}

func (repo *turnStateSettingsTestRepo) GetValue(context.Context, string) (string, error) {
	repo.mu.Lock()
	defer repo.mu.Unlock()
	if repo.value == "" {
		return "", ErrSettingNotFound
	}
	return repo.value, nil
}

func (repo *turnStateSettingsTestRepo) Set(_ context.Context, _ string, value string) error {
	repo.mu.Lock()
	defer repo.mu.Unlock()
	if repo.fail {
		return errors.New("test storage failure")
	}
	repo.value = value
	return nil
}

func turnStateSettingsTestGateway(test *testing.T, repo *turnStateSettingsTestRepo) *OpenAIGatewayService {
	cache := redis.NewClient(&redis.Options{Addr: miniredis.RunT(test).Addr()})
	gateway := &OpenAIGatewayService{cfg: &config.Config{JWT: config.JWTConfig{Secret: "test-stable-server-secret"}}, settingService: NewSettingService(repo, nil)}
	gateway.turnStateAuto = turnstate.NewRuntime(func(settings turnstate.Config) (*turnstate.Manager, error) {
		return turnstate.New(settings, cache, nil)
	})
	gateway.initializeTurnStateSettings(turnstate.Config{Attempts: 9, Concurrency: 4})
	test.Cleanup(func() { gateway.CloseTurnStateAuto(); _ = cache.Close() })
	return gateway
}

func TestTurnStateSettingsEncryptedAndHotApplied(test *testing.T) {
	repo := &turnStateSettingsTestRepo{}
	gateway := turnStateSettingsTestGateway(test, repo)
	ctx := context.Background()
	view, err := gateway.GetTurnStateSettings(ctx)
	require.NoError(test, err)
	require.False(test, view.PurchasedEnabled)
	require.Equal(test, 48, view.RefreshAfterMinutes)
	require.Equal(test, []string{"codex-auto-review", "gpt-5.6-terra", "gpt-5.4"}, view.ExcludedModels)
	view.PurchasedEnabled = true
	view.ProxyHost, view.ProxyUsername, view.ProxyPassword = "proxy.example:7778", "account_{country}_{session}", "private-proxy-password"
	view.ProxyUpstream = "socks5://user:upstream-secret@localhost:7897"
	view.IPv6Enabled, view.PoolURL, view.PoolToken = true, "https://pool.example:18443", strings.Repeat("pool-secret", 4)
	saved, err := gateway.SaveTurnStateSettings(ctx, view)
	require.NoError(test, err)
	require.True(test, gateway.turnStateAuto.Configured(turnstate.SourcePurchased))
	require.True(test, gateway.turnStateAuto.Configured(turnstate.SourceIPv6))
	require.True(test, saved.ProxyPasswordConfigured)
	require.True(test, saved.PoolTokenConfigured)
	require.True(test, saved.ProxyUpstreamConfigured)
	encoded, err := json.Marshal(saved)
	require.NoError(test, err)
	for _, secret := range []string{view.ProxyPassword, view.PoolToken, "upstream-secret"} {
		require.NotContains(test, string(encoded), secret)
		require.NotContains(test, repo.value, secret)
	}
	saved.Countries = "US,DE"
	saved.RefreshAfterMinutes = 25
	saved.ExcludedModels = []string{}
	saved, err = gateway.SaveTurnStateSettings(ctx, saved)
	require.NoError(test, err)
	require.True(test, saved.ProxyPasswordConfigured)
	require.Equal(test, []string{"US", "DE"}, gateway.turnStateAuto.Countries())
	restored := turnStateSettingsTestGateway(test, repo)
	require.True(test, restored.turnStateAuto.Configured(turnstate.SourcePurchased))
	restoredView, err := restored.GetTurnStateSettings(ctx)
	require.NoError(test, err)
	require.Equal(test, 25, restoredView.RefreshAfterMinutes)
	require.NotNil(test, restoredView.ExcludedModels)
	require.Empty(test, restoredView.ExcludedModels)
	require.False(test, restored.turnStateAuto.ModelExcluded("gpt-5.6-terra"))
	saved.ClearProxyUpstream = true
	saved, err = gateway.SaveTurnStateSettings(ctx, saved)
	require.NoError(test, err)
	require.False(test, saved.ProxyUpstreamConfigured)
	saved.PurchasedEnabled = false
	_, err = gateway.SaveTurnStateSettings(ctx, saved)
	require.NoError(test, err)
	restored.turnStateSettings.sync(ctx)
	require.False(test, restored.turnStateAuto.Configured(turnstate.SourcePurchased))
	require.True(test, restored.turnStateAuto.Configured(turnstate.SourceIPv6))
}

func TestTurnStateSettingsRejectsInvalidAndPreservesPrevious(test *testing.T) {
	repo := &turnStateSettingsTestRepo{}
	gateway := turnStateSettingsTestGateway(test, repo)
	ctx := context.Background()
	view, err := gateway.GetTurnStateSettings(ctx)
	require.NoError(test, err)
	view.PurchasedEnabled = true
	_, err = gateway.SaveTurnStateSettings(ctx, view)
	require.Error(test, err)
	require.Empty(test, repo.value)
	view.ProxyHost, view.ProxyUsername, view.ProxyPassword = "proxy.example:7778", "account_{country}_{session}", "secret"
	first, err := gateway.SaveTurnStateSettings(ctx, view)
	require.NoError(test, err)
	_, err = gateway.SaveTurnStateSettings(ctx, view)
	require.ErrorContains(test, err, "其他管理员")
	oldValue := repo.value
	invalid := first
	invalid.ProxyHost = "not-a-host-port"
	_, err = gateway.SaveTurnStateSettings(ctx, invalid)
	require.Error(test, err)
	require.Equal(test, oldValue, repo.value)
	invalid = first
	invalid.RefreshAfterMinutes = 60
	_, err = gateway.SaveTurnStateSettings(ctx, invalid)
	require.ErrorContains(test, err, "1–59")
	require.Equal(test, oldValue, repo.value)
	invalid = first
	invalid.ExcludedModels = []string{"bad\nmodel"}
	_, err = gateway.SaveTurnStateSettings(ctx, invalid)
	require.ErrorContains(test, err, "完整模型名称")
	require.Equal(test, oldValue, repo.value)
	first.Countries = "DE"
	repo.mu.Lock()
	repo.fail = true
	repo.mu.Unlock()
	_, err = gateway.SaveTurnStateSettings(ctx, first)
	require.ErrorContains(test, err, "原配置保持不变")
	require.Equal(test, oldValue, repo.value)
	require.True(test, gateway.turnStateAuto.Configured(turnstate.SourcePurchased))
	require.NotEqual(test, []string{"DE"}, gateway.turnStateAuto.Countries())
}
