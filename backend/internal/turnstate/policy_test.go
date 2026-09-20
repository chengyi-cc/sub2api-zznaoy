package turnstate

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestIncludedModelsMigrationEmptyAndCustomPolicy(test *testing.T) {
	var legacy Config
	require.NoError(test, json.Unmarshal([]byte(`{"ExcludedModels":["gpt-6-astra"],"RefreshAfterMinutes":25}`), &legacy))
	config, err := NormalizeAcquisitionPolicy(legacy)
	require.NoError(test, err)
	require.Equal(test, strings.Split(DefaultIncludedModels, ","), config.IncludedModels)
	require.Equal(test, 25, config.RefreshAfterMinutes)
	runtime := NewRuntime(func(Config) (*Manager, error) { return nil, nil })
	defer runtime.Close()
	require.NoError(test, runtime.Update(config, nil))
	for _, model := range strings.Split(DefaultIncludedModels, ",") {
		require.False(test, runtime.ModelExcluded(model))
		require.True(test, runtime.RequiresValidState(model))
	}
	require.True(test, runtime.ModelExcluded("future-model"))
	require.False(test, runtime.RequiresValidState("future-model"))
	config.IncludedModels = []string{}
	require.NoError(test, runtime.Update(config, nil))
	require.True(test, runtime.ModelExcluded("gpt-6-astra"))
	require.False(test, runtime.RequiresValidState("gpt-6-astra"))
	config.IncludedModels = []string{" Future-Model ", "future-model"}
	config, err = NormalizeAcquisitionPolicy(config)
	require.NoError(test, err)
	require.Equal(test, []string{"future-model"}, config.IncludedModels)
	require.NoError(test, runtime.Update(config, nil))
	require.False(test, runtime.ModelExcluded(" FUTURE-MODEL "))
	require.True(test, runtime.RequiresValidState("future-model"))
	require.True(test, runtime.ModelExcluded("future-model-mini"))
}

func TestIncludedModelsEnvironmentDefaultsAndEmpty(test *testing.T) {
	test.Setenv("TURN_STATE_EXCLUDED_MODELS", "gpt-6-astra")
	test.Setenv("TURN_STATE_INCLUDED_MODELS", "gpt-5.5,custom-model")
	config, err := NormalizeAcquisitionPolicy(ConfigFromEnv())
	require.NoError(test, err)
	require.Equal(test, []string{"gpt-5.5", "custom-model"}, config.IncludedModels)
	test.Setenv("TURN_STATE_INCLUDED_MODELS", "")
	config, err = NormalizeAcquisitionPolicy(ConfigFromEnv())
	require.NoError(test, err)
	require.NotNil(test, config.IncludedModels)
	require.Empty(test, config.IncludedModels)
}

func TestUnlistedModelsSkipCacheInjectionAndAllAcquisition(test *testing.T) {
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
	require.True(test, manager.ModelExcluded("gpt-5.4-mini"))
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
	config.IncludedModels = []string{"gpt-5.6-terra", "gpt-6-astra"}
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
	config.IncludedModels = nil
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
	config.IncludedModels = []string{"gpt-5.6-terra", "gpt-6-astra"}
	config.RefreshAfterMinutes = 55
	require.NoError(test, runtime.Update(config, nil))
	require.False(test, runtime.ModelExcluded("gpt-5.6-terra"))
	require.True(test, runtime.Force(ctx, 42, "gpt-5.6-terra", make(http.Header), Options{}))
}

func TestAcquisitionPolicyDefaultsAndValidation(test *testing.T) {
	config, err := NormalizeAcquisitionPolicy(Config{})
	require.NoError(test, err)
	require.Equal(test, 48, config.RefreshAfterMinutes)
	require.Equal(test, []string{"gpt-6-astra", "gpt-5.6-sol", "gpt-5.5"}, config.IncludedModels)
	config, err = NormalizeAcquisitionPolicy(Config{RefreshAfterMinutes: 25, IncludedModels: []string{}})
	require.NoError(test, err)
	require.Empty(test, config.IncludedModels)
	require.NotNil(test, config.IncludedModels)
	for _, invalid := range []Config{{RefreshAfterMinutes: -1}, {RefreshAfterMinutes: 60}, {IncludedModels: []string{"bad\nmodel"}}, {IncludedModels: []string{"gpt-*"}}} {
		_, err = NormalizeAcquisitionPolicy(invalid)
		require.Error(test, err)
	}
}
