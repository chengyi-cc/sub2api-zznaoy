package turnstate

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestExcludedModelsSkipCacheInjectionAndAllAcquisition(test *testing.T) {
	manager, cache := managerForTest(test)
	ctx := context.Background()
	for _, model := range []string{"codex-auto-review", "gpt-5.6-terra", "gpt-5.4"} {
		storeRecord(test, cache, 42, model, stateValue(time.Now(), 249))
		require.NoError(test, cache.SAdd(ctx, accountIndexKey(42), model).Err())
		headers := make(http.Header)
		headers.Set(Header, "client-state")
		require.False(test, manager.Apply(ctx, 42, model, headers))
		require.False(test, manager.Force(ctx, 42, model, headers, Options{}))
		require.Equal(test, "client-state", headers.Get(Header))
	}
	require.Empty(test, manager.Snapshot(42))
	require.Empty(test, manager.Inspect(ctx, 42, Options{}))
	require.Empty(test, manager.slots)
	require.True(test, manager.ModelExcluded(" GPT-5.6-TERRA "))
	require.False(test, manager.ModelExcluded("gpt-5.4-mini"))
	require.False(test, manager.ModelExcluded("gpt-6-astra"))
}

func TestRuntimePolicyUpdateCancelsExcludedWorkAndRecalculatesRefresh(test *testing.T) {
	original, cache := managerForTest(test)
	started := make(chan struct{}, 4)
	runtime := NewRuntime(func(config Config) (*Manager, error) {
		manager, err := New(config, cache, nil)
		if err == nil && manager != nil {
			manager.sample = func(ctx context.Context, _ http.Header, _ string) (Record, error) {
				started <- struct{}{}
				<-ctx.Done()
				return Record{}, ctx.Err()
			}
		}
		return manager, err
	})
	test.Cleanup(runtime.Close)
	config := original.config
	config.ExcludedModels = []string{}
	require.NoError(test, runtime.Update(config, nil))
	ctx := context.Background()
	require.True(test, runtime.Force(ctx, 42, "gpt-5.6-terra", make(http.Header), Options{}))
	select {
	case <-started:
	case <-time.After(time.Second):
		test.Fatal("acquisition did not start")
	}
	storeRecord(test, cache, 42, "gpt-6-astra", stateValue(time.Now().Add(-40*time.Minute), 249))
	require.True(test, runtime.Apply(ctx, 42, "gpt-6-astra", make(http.Header)))
	previous := runtime.manager
	config.ExcludedModels = nil
	config.RefreshAfterMinutes = 55
	require.NoError(test, runtime.Update(config, nil))
	require.Empty(test, previous.slots)
	require.True(test, runtime.ModelExcluded("gpt-5.6-terra"))
	require.False(test, runtime.Force(ctx, 42, "gpt-5.6-terra", make(http.Header), Options{}))
	statuses := runtime.Inspect(ctx, 42, Options{})
	require.Len(test, statuses, 1)
	require.Equal(test, "gpt-6-astra", statuses[0].Model)
	require.Equal(test, 55*time.Minute, statuses[0].RefreshAt.Sub(*statuses[0].IssuedAt))
	require.Equal(test, time.Hour, statuses[0].ExpiresAt.Sub(*statuses[0].IssuedAt))
	require.Empty(test, runtime.manager.slots)
	config.RefreshAfterMinutes = 20
	require.NoError(test, runtime.Update(config, nil))
	select {
	case <-started:
	case <-time.After(time.Second):
		test.Fatal("earlier refresh age did not start existing target")
	}
	config.ExcludedModels = []string{}
	config.RefreshAfterMinutes = 55
	require.NoError(test, runtime.Update(config, nil))
	require.False(test, runtime.ModelExcluded("gpt-5.6-terra"))
	require.True(test, runtime.Force(ctx, 42, "gpt-5.6-terra", make(http.Header), Options{}))
}

func TestAcquisitionPolicyDefaultsAndValidation(test *testing.T) {
	config, err := NormalizeAcquisitionPolicy(Config{})
	require.NoError(test, err)
	require.Equal(test, 48, config.RefreshAfterMinutes)
	require.Equal(test, []string{"codex-auto-review", "gpt-5.6-terra", "gpt-5.4"}, config.ExcludedModels)
	config, err = NormalizeAcquisitionPolicy(Config{RefreshAfterMinutes: 25, ExcludedModels: []string{}})
	require.NoError(test, err)
	require.Empty(test, config.ExcludedModels)
	require.NotNil(test, config.ExcludedModels)
	for _, invalid := range []Config{{RefreshAfterMinutes: -1}, {RefreshAfterMinutes: 60}, {ExcludedModels: []string{"bad\nmodel"}}, {ExcludedModels: []string{"gpt-*"}}} {
		_, err = NormalizeAcquisitionPolicy(invalid)
		require.Error(test, err)
	}
}
