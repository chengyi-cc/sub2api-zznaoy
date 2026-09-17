package turnstate

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestPendingTargetsDoNotRestartFailedAcquisitionDuringCooldown(test *testing.T) {
	manager, _ := managerForTest(test)
	manager.config.Attempts = 1
	manager.slots = make(chan struct{}, 1)
	var calls atomic.Int64
	manager.sample = func(context.Context, http.Header, string) (Record, error) {
		calls.Add(1)
		return Record{}, errors.New("sample rejected")
	}
	for accountID := int64(1); accountID <= 2; accountID++ {
		manager.Apply(context.Background(), accountID, "model-a", make(http.Header))
	}
	require.Eventually(test, func() bool {
		for accountID := int64(1); accountID <= 2; accountID++ {
			statuses := manager.Snapshot(accountID)
			if len(statuses) != 1 || statuses[0].State != "unavailable" {
				return false
			}
		}
		return true
	}, 4*time.Second, 10*time.Millisecond)
	manager.mu.Lock()
	manager.schedulePendingLocked()
	manager.mu.Unlock()
	require.EqualValues(test, 2, calls.Load())
}

func TestCloseDoesNotStartPendingTargets(test *testing.T) {
	manager, _ := managerForTest(test)
	started := make(chan struct{}, 4)
	var calls atomic.Int64
	manager.sample = func(ctx context.Context, _ http.Header, _ string) (Record, error) {
		calls.Add(1)
		started <- struct{}{}
		<-ctx.Done()
		return Record{}, ctx.Err()
	}
	for accountID := int64(1); accountID <= 48; accountID++ {
		manager.Apply(context.Background(), accountID, "model-a", make(http.Header))
	}
	for worker := 0; worker < 4; worker++ {
		select {
		case <-started:
		case <-time.After(time.Second):
			test.Fatal("worker did not start")
		}
	}
	manager.Close()
	require.EqualValues(test, 4, calls.Load())
	require.Empty(test, manager.slots)
}

func TestManyTargetsDrainWithoutWaitingForRefreshTicker(test *testing.T) {
	manager, client := managerForTest(test)
	var active atomic.Int64
	var maximum atomic.Int64
	var calls atomic.Int64
	started := make(chan struct{}, 4)
	proceed := make(chan struct{})
	var release sync.Once
	defer release.Do(func() { close(proceed) })
	manager.sample = func(ctx context.Context, headers http.Header, model string) (Record, error) {
		current := active.Add(1)
		defer active.Add(-1)
		calls.Add(1)
		for previous := maximum.Load(); current > previous; previous = maximum.Load() {
			if maximum.CompareAndSwap(previous, current) {
				break
			}
		}
		select {
		case started <- struct{}{}:
		default:
		}
		select {
		case <-proceed:
		case <-ctx.Done():
			return Record{}, ctx.Err()
		}
		time.Sleep(10 * time.Millisecond)
		return Parse(stateValue(time.Now(), 249), model, time.Now())
	}
	for accountID := int64(1); accountID <= 48; accountID++ {
		for _, model := range []string{"model-a", "model-b"} {
			headers := make(http.Header)
			headers.Set("Chatgpt-Account-Id", strconv.FormatInt(accountID, 10))
			require.False(test, manager.Apply(context.Background(), accountID, model, headers))
		}
	}
	for worker := 0; worker < 4; worker++ {
		select {
		case <-started:
		case <-time.After(time.Second):
			test.Fatal("worker did not start")
		}
	}
	release.Do(func() { close(proceed) })
	require.Eventually(test, func() bool {
		for accountID := int64(1); accountID <= 48; accountID++ {
			for _, model := range []string{"model-a", "model-b"} {
				if client.Exists(context.Background(), recordKey(accountID, model)).Val() != 1 {
					return false
				}
			}
		}
		return true
	}, 5*time.Second, 20*time.Millisecond)
	require.EqualValues(test, 96, calls.Load())
	require.EqualValues(test, 4, maximum.Load())
	for accountID := int64(1); accountID <= 48; accountID++ {
		for _, model := range []string{"model-a", "model-b"} {
			headers := make(http.Header)
			headers.Set("Chatgpt-Account-Id", strconv.FormatInt(accountID, 10))
			require.True(test, manager.Apply(context.Background(), accountID, model, headers))
			require.Len(test, headers.Get(Header), 332)
		}
	}
}
