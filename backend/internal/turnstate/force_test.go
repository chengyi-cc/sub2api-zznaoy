package turnstate

import (
	"context"
	"errors"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestForceBypassesFreshCacheAndKeepsOldStateOnFailure(test *testing.T) {
	manager, cache := managerForTest(test)
	manager.config.Attempts = 1
	var calls atomic.Int32
	manager.sample = func(context.Context, http.Header, string) (Record, error) {
		calls.Add(1)
		return Record{}, errors.New("test probe failure")
	}
	old := stateValue(time.Now(), 249)
	storeRecord(test, cache, 42, "model", old)
	require.True(test, manager.Force(context.Background(), 42, "model", make(http.Header), Options{}))
	require.Eventually(test, func() bool {
		manager.mu.Lock()
		defer manager.mu.Unlock()
		return !manager.targets[recordKey(42, "model")].running && calls.Load() == 1
	}, 3*time.Second, 10*time.Millisecond)
	headers := make(http.Header)
	require.True(test, manager.Apply(context.Background(), 42, "model", headers))
	require.Equal(test, old, headers.Get(Header))
	history, err := manager.History(context.Background(), 42)
	require.NoError(test, err)
	require.Len(test, history, 1)
	require.False(test, history[0].Accepted)
}

func TestForceQueuesFreshModelAndDeduplicatesRunningTask(test *testing.T) {
	manager, cache := managerForTest(test)
	manager.slots = make(chan struct{}, 1)
	started := make(chan string, 3)
	release := make(chan struct{})
	manager.sample = func(ctx context.Context, _ http.Header, model string) (Record, error) {
		started <- model
		select {
		case <-release:
		case <-ctx.Done():
			return Record{}, ctx.Err()
		}
		return Parse(stateValue(time.Now(), 249), model, time.Now())
	}
	require.True(test, manager.Force(context.Background(), 42, "first", make(http.Header), Options{}))
	select {
	case <-started:
	case <-time.After(time.Second):
		test.Fatal("initial acquisition did not start")
	}
	require.True(test, manager.Force(context.Background(), 42, "first", make(http.Header), Options{}))
	storeRecord(test, cache, 42, "second", stateValue(time.Now(), 249))
	require.True(test, manager.Force(context.Background(), 42, "second", make(http.Header), Options{}))
	manager.mu.Lock()
	require.True(test, manager.targets[recordKey(42, "second")].force)
	require.Equal(test, "queued", manager.targets[recordKey(42, "second")].status.State)
	manager.mu.Unlock()
	close(release)
	select {
	case model := <-started:
		require.Equal(test, "second", model)
	case <-time.After(2 * time.Second):
		test.Fatal("queued fresh-cache acquisition did not start")
	}
	require.Eventually(test, func() bool {
		manager.mu.Lock()
		defer manager.mu.Unlock()
		return !manager.targets[recordKey(42, "second")].running
	}, time.Second, 10*time.Millisecond)
	require.Empty(test, started)
	manager.Close()
	require.False(test, manager.Force(context.Background(), 42, "second", make(http.Header), Options{}))
}
