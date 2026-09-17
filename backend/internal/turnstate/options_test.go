package turnstate

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestTeamAndProRulesRemainIndependent(test *testing.T) {
	now := time.Now()
	for _, candidate := range []struct {
		profile   string
		rawLength int
		accepted  bool
	}{
		{ProfileTeam, 249, true}, {ProfileTeam, 265, false}, {ProfileTeam, 217, false},
		{ProfilePro, 217, true}, {ProfilePro, 233, false}, {ProfilePro, 249, false},
	} {
		_, err := Parse(stateValue(now, candidate.rawLength), "model", now, candidate.profile)
		require.Equal(test, candidate.accepted, err == nil)
	}
	require.Equal(test, Options{Profile: ProfileTeam, Source: SourcePurchased}, OptionsFromExtra(nil))
	require.Equal(test, Options{Profile: ProfilePro, Source: SourceIPv6}, OptionsFromExtra(map[string]any{ProfileKey: ProfilePro, SourceKey: SourceIPv6}))
	require.Error(test, ValidateOptions(map[string]any{ProfileKey: "unknown"}))
	require.Error(test, ValidateOptions(map[string]any{SourceKey: true}))
}

func TestLegacyProCacheIsAvailableOnlyWhenExplicitlySelected(test *testing.T) {
	manager, client := managerForTest(test)
	manager.sample = func(context.Context, http.Header, string) (Record, error) {
		return Record{}, errors.New("test sampling disabled")
	}
	now := time.Now()
	legacy := map[string]any{"value": stateValue(now, 217), "model": "model", "issued_at": now, "expires_at": now.Add(time.Hour), "acquired_at": now}
	data, err := json.Marshal(legacy)
	require.NoError(test, err)
	require.NoError(test, client.Set(context.Background(), recordKey(5, "model"), data, time.Hour).Err())
	headers := make(http.Header)
	require.True(test, manager.Apply(context.Background(), 5, "model", headers, Options{Profile: ProfilePro, Source: SourceIPv6}))
	require.Len(test, headers.Get(Header), 292)
	teamHeaders := make(http.Header)
	require.False(test, manager.Apply(context.Background(), 5, "model", teamHeaders))
	require.Empty(test, teamHeaders.Get(Header))
}

func TestRefreshStartsAtThirtyMinutesAndKeepsValidOldState(test *testing.T) {
	manager, client := managerForTest(test)
	manager.config.Attempts = 1
	var calls atomic.Int64
	manager.sample = func(context.Context, http.Header, string) (Record, error) {
		calls.Add(1)
		return Record{}, errors.New("no accepted candidate")
	}
	value := stateValue(time.Now().Add(-31*time.Minute), 249)
	storeRecord(test, client, 42, "model", value)
	headers := make(http.Header)
	require.True(test, manager.Apply(context.Background(), 42, "model", headers))
	require.Equal(test, value, headers.Get(Header))
	require.Eventually(test, func() bool {
		statuses := manager.Snapshot(42)
		return len(statuses) == 1 && statuses[0].LastError == "no accepted candidate"
	}, 3*time.Second, 10*time.Millisecond)
	require.EqualValues(test, 1, calls.Load())
	statuses := manager.Snapshot(42)
	require.Equal(test, "ready", statuses[0].State)
	require.InDelta(test, 30*time.Minute, statuses[0].ExpiresAt.Sub(*statuses[0].RefreshAt), float64(time.Second))
	record, err := manager.read(context.Background(), 42, "model")
	require.NoError(test, err)
	require.Equal(test, value, record.Value)
	storeRecord(test, client, 43, "model", stateValue(time.Now().Add(-29*time.Minute), 249))
	require.True(test, manager.Apply(context.Background(), 43, "model", make(http.Header)))
	require.EqualValues(test, 1, calls.Load())
}

func TestCountryRotatesAfterThreeFailuresAcrossRoundsAndHistoryPersists(test *testing.T) {
	manager, client := managerForTest(test)
	manager.config.Countries = []string{"US", "JP"}
	manager.config.Attempts = 2
	var mutex sync.Mutex
	countries := []string{}
	manager.sample = func(ctx context.Context, _ http.Header, model string) (Record, error) {
		mutex.Lock()
		countries = append(countries, sampleOptions(ctx).Country)
		count := len(countries)
		mutex.Unlock()
		if count <= 3 {
			return Record{}, &ProbeError{Message: "candidate length 356 is not accepted", Length: 356, Status: 200, SourceIP: "198.51.100.1", Country: "US"}
		}
		record, err := Parse(stateValue(time.Now(), 249), model, time.Now())
		record.Country, record.SourceIP = "JP", "198.51.100.2"
		return record, err
	}
	manager.Apply(context.Background(), 7, "model", make(http.Header))
	require.Eventually(test, func() bool {
		statuses := manager.Snapshot(7)
		return len(statuses) == 1 && statuses[0].State == "unavailable"
	}, 4*time.Second, 10*time.Millisecond)
	rotation, err := manager.readRotation(context.Background(), recordKey(7, "model"), Options{}.Normalized())
	require.NoError(test, err)
	require.Equal(test, 2, rotation.Failures)
	manager.mu.Lock()
	manager.targets[recordKey(7, "model")].retryAt = time.Time{}
	manager.schedulePendingLocked()
	manager.mu.Unlock()
	require.Eventually(test, func() bool { return client.Exists(context.Background(), recordKey(7, "model")).Val() == 1 }, 3*time.Second, 10*time.Millisecond)
	mutex.Lock()
	require.Equal(test, []string{"US", "US", "US", "JP"}, countries)
	mutex.Unlock()
	rotation, err = manager.readRotation(context.Background(), recordKey(7, "model"), Options{}.Normalized())
	require.NoError(test, err)
	require.Equal(test, countryRotation{Index: 1}, rotation)
	history, err := manager.History(context.Background(), 7)
	require.NoError(test, err)
	require.Len(test, history, 4)
	require.True(test, history[0].Accepted)
	require.Equal(test, 332, history[0].Length)
	require.Equal(test, "JP", history[0].Country)
	encoded, _ := json.Marshal(history)
	record, err := manager.read(context.Background(), 7, "model")
	require.NoError(test, err)
	require.NotContains(test, string(encoded), record.Value)
	other, err := New(manager.config, client, nil)
	require.NoError(test, err)
	defer other.Close()
	reloaded, err := other.History(context.Background(), 7)
	require.NoError(test, err)
	require.Len(test, reloaded, 4)
	statuses := other.Inspect(context.Background(), 7, Options{}.Normalized())
	require.Len(test, statuses, 1)
	require.Equal(test, 332, statuses[0].Length)
}

func TestChangingProfileDuringAcquisitionDoesNotPublish(test *testing.T) {
	manager, client := managerForTest(test)
	var checks atomic.Int64
	manager.prepare = func(_ context.Context, _ int64, headers http.Header) (http.Header, Options, bool) {
		options := Options{}.Normalized()
		if checks.Add(1) > 1 {
			options.Profile = ProfilePro
		}
		return headers, options, true
	}
	manager.sample = func(_ context.Context, _ http.Header, model string) (Record, error) {
		return Parse(stateValue(time.Now(), 249), model, time.Now())
	}
	manager.Apply(context.Background(), 1, "model", make(http.Header))
	require.Eventually(test, func() bool {
		statuses := manager.Snapshot(1)
		return len(statuses) == 1 && statuses[0].LastError == "account disabled during acquisition"
	}, time.Second, 10*time.Millisecond)
	require.Zero(test, client.Exists(context.Background(), recordKey(1, "model")).Val())
}
