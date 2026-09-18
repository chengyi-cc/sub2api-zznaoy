package turnstate

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestResponseInvalidationCoalescesConcurrentRepliesAndProtectsReplacement(test *testing.T) {
	for _, profile := range []string{ProfilePro, ProfileTeam} {
		test.Run(profile, func(test *testing.T) {
			manager, cache := managerForTest(test)
			options := Options{Profile: profile, Source: SourcePurchased}
			rawLength, rejectionLength := 249, 356
			if profile == ProfilePro {
				rawLength, rejectionLength = 217, 312
			}
			ctx := context.Background()
			old, err := Parse(stateValue(time.Now().Add(-5*time.Minute), rawLength), "model", time.Now(), profile)
			require.NoError(test, err)
			old.Options = options
			require.NoError(test, manager.saveAcquiredRecord(ctx, 42, old))
			proceed := make(chan struct{})
			var release sync.Once
			defer release.Do(func() { close(proceed) })
			var samples atomic.Int64
			manager.sample = func(ctx context.Context, _ http.Header, model string) (Record, error) {
				samples.Add(1)
				select {
				case <-proceed:
					return Parse(stateValue(time.Now(), rawLength), model, time.Now(), profile)
				case <-ctx.Done():
					return Record{}, ctx.Err()
				}
			}
			sent, received := make(http.Header), make(http.Header)
			require.True(test, manager.Apply(ctx, 42, "model", sent, options))
			received.Set(Header, strings.Repeat("x", rejectionLength))
			var changed atomic.Int64
			var requests sync.WaitGroup
			for request := 0; request < 20; request++ {
				requests.Add(1)
				go func() {
					defer requests.Done()
					if manager.ObserveResponse(ctx, 42, "model", sent, received, 200, options) {
						changed.Add(1)
					}
				}()
			}
			requests.Wait()
			require.EqualValues(test, 1, changed.Load())
			require.Eventually(test, func() bool { return samples.Load() == 1 }, time.Second, time.Millisecond)
			_, err = manager.read(ctx, 42, "model")
			require.ErrorIs(test, err, errInvalidated)
			headers := sent.Clone()
			require.False(test, manager.Apply(ctx, 42, "model", headers, options))
			require.Empty(test, headers.Get(Header))
			require.ErrorContains(test, manager.saveAcquiredRecord(ctx, 42, old), "invalidated")
			second, err := New(manager.config, cache, nil)
			require.NoError(test, err)
			defer second.Close()
			_, err = second.read(ctx, 42, "model")
			require.ErrorIs(test, err, errInvalidated)
			history, err := manager.History(ctx, 42)
			require.NoError(test, err)
			require.Len(test, history, 1)
			require.Equal(test, "response_rejection", history[0].Kind)
			release.Do(func() { close(proceed) })
			require.Eventually(test, func() bool {
				replacement, readErr := manager.read(ctx, 42, "model")
				return readErr == nil && replacement.Value != old.Value
			}, time.Second, time.Millisecond)
			replacement, err := manager.read(ctx, 42, "model")
			require.NoError(test, err)
			require.False(test, manager.ObserveResponse(ctx, 42, "model", sent, received, 200, options))
			stillCurrent, err := manager.read(ctx, 42, "model")
			require.NoError(test, err)
			require.Equal(test, replacement.Value, stillCurrent.Value)
			require.EqualValues(test, 1, samples.Load())
		})
	}
}

func TestResponseInvalidationIgnoresUnrelatedSignalsAndDisabledPolicy(test *testing.T) {
	manager, cache := managerForTest(test)
	ctx := context.Background()
	storeRecord(test, cache, 42, "model", stateValue(time.Now(), 249))
	sent := make(http.Header)
	require.True(test, manager.Apply(ctx, 42, "model", sent))
	received := make(http.Header)
	for _, length := range []int{0, 312, 332, 376} {
		received.Set(Header, strings.Repeat("x", length))
		require.False(test, manager.ObserveResponse(ctx, 42, "model", sent, received, 200, Options{}))
	}
	received.Set(Header, strings.Repeat("x", 356))
	require.False(test, manager.ObserveResponse(ctx, 43, "model", sent, received, 200, Options{}))
	require.False(test, manager.ObserveResponse(ctx, 42, "different-model", sent, received, 200, Options{}))
	otherIdentity := sent.Clone()
	otherIdentity.Set("Chatgpt-Account-Id", "other-account")
	require.False(test, manager.ObserveResponse(ctx, 42, "model", otherIdentity, received, 200, Options{}))
	wrongSource := Options{Profile: ProfileTeam, Source: SourceIPv6}
	require.False(test, manager.ObserveResponse(ctx, 42, "model", sent, received, 200, wrongSource))
	config := manager.config
	disabled := false
	config.RefreshOnRejection = &disabled
	disabledManager, err := New(config, cache, nil)
	require.NoError(test, err)
	defer disabledManager.Close()
	require.False(test, disabledManager.ObserveResponse(ctx, 42, "model", sent, received, 200, Options{}))
	_, err = manager.read(ctx, 42, "model")
	require.NoError(test, err)
	stored, err := cache.Get(ctx, recordKey(42, "model")).Bytes()
	require.NoError(test, err)
	var record Record
	require.NoError(test, json.Unmarshal(stored, &record))
	require.False(test, record.Invalidated)
}
