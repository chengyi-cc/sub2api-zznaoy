package turnstate

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
)

func stateValue(issued time.Time, length int) string {
	raw := make([]byte, length)
	raw[0] = 0x80
	binary.BigEndian.PutUint64(raw[1:9], uint64(issued.Unix()))
	return base64.URLEncoding.EncodeToString(raw)
}

func TestParseLengthAndLifetime(test *testing.T) {
	now := time.Now().UTC().Truncate(time.Second)
	normal := stateValue(now.Add(-40*time.Minute), 217)
	require.Len(test, normal, 292)
	record, err := Parse(normal, "model-a", now)
	require.NoError(test, err)
	require.Equal(test, now.Add(20*time.Minute), record.ExpiresAt)
	abnormal := stateValue(now, 233)
	require.Len(test, abnormal, 312)
	for _, value := range []string{abnormal, "", strings.Repeat("x", 292), stateValue(now.Add(-time.Hour), 217), stateValue(now.Add(time.Minute), 217), normal[:291], normal[:100] + "\n" + normal[101:]} {
		_, err := Parse(value, "model-a", now)
		require.Error(test, err)
	}
	_, err = Parse("  "+normal+"  ", "model-a", now)
	require.NoError(test, err)
}

func TestParseRejectsWrongCiphertextBlockLength(test *testing.T) {
	value := stateValue(time.Now(), 218)
	require.Len(test, value, 292)
	_, err := Parse(value, "model-a", time.Now())
	require.ErrorContains(test, err, "format")
}

func TestKnownUpstreamIdentityChangeDoesNotReuseState(test *testing.T) {
	manager, client := managerForTest(test)
	manager.sample = func(context.Context, http.Header, string) (Record, error) {
		return Record{}, errors.New("test disabled")
	}
	value := stateValue(time.Now(), 217)
	storeRecord(test, client, 42, "model-a", value)
	headers := make(http.Header)
	headers.Set("Chatgpt-Account-Id", "new-upstream-account")
	headers.Set(Header, "client-state")
	require.False(test, manager.Apply(context.Background(), 42, "model-a", headers))
	require.Equal(test, "client-state", headers.Get(Header))
}

func managerForTest(test *testing.T) (*Manager, *redis.Client) {
	server := miniredis.RunT(test)
	client := redis.NewClient(&redis.Options{Addr: server.Addr()})
	manager, err := New(Config{URL: "https://pool.example:18443", Token: strings.Repeat("a", 48), Attempts: 2, Concurrency: 4}, client, nil)
	require.NoError(test, err)
	test.Cleanup(func() { manager.Close(); _ = client.Close() })
	return manager, client
}

func storeRecord(test *testing.T, client *redis.Client, accountID int64, model, value string) {
	record, err := Parse(value, model, time.Now())
	require.NoError(test, err)
	encoded, err := json.Marshal(record)
	require.NoError(test, err)
	require.NoError(test, client.Set(context.Background(), recordKey(accountID, model), encoded, time.Until(record.ExpiresAt)).Err())
}

func TestApplyOverridesButNeverCrossesAccountOrModel(test *testing.T) {
	manager, client := managerForTest(test)
	manager.sample = func(context.Context, http.Header, string) (Record, error) {
		return Record{}, errors.New("test disabled")
	}
	value := stateValue(time.Now(), 217)
	storeRecord(test, client, 42, "model-a", value)
	headers := http.Header{Header: []string{"old-value"}}
	require.True(test, manager.Apply(context.Background(), 42, "model-a", headers))
	require.Equal(test, value, headers.Get(Header))
	for _, target := range []struct {
		account int64
		model   string
	}{{43, "model-a"}, {42, "model-b"}, {42, ""}} {
		untouched := make(http.Header)
		untouched.Set(Header, "client-value")
		require.False(test, manager.Apply(context.Background(), target.account, target.model, untouched))
		require.Equal(test, "client-value", untouched.Get(Header))
	}
	require.Len(test, manager.Snapshot(42), 2)
	encoded, err := json.Marshal(manager.Snapshot(42))
	require.NoError(test, err)
	require.NotContains(test, string(encoded), value)
}

func TestRefreshDeduplicatedAndPersists(test *testing.T) {
	manager, client := managerForTest(test)
	var calls atomic.Int64
	started := make(chan struct{})
	proceed := make(chan struct{})
	manager.sample = func(ctx context.Context, headers http.Header, model string) (Record, error) {
		calls.Add(1)
		require.Empty(test, headers.Get(Header))
		require.Empty(test, headers.Get("Cookie"))
		close(started)
		select {
		case <-ctx.Done():
			return Record{}, ctx.Err()
		case <-proceed:
		}
		return Parse(stateValue(time.Now(), 217), model, time.Now())
	}
	headers := make(http.Header)
	headers.Set(Header, "old-state")
	headers.Set("Cookie", "do-not-copy")
	headers.Set("Authorization", "Bearer secret")
	var requests sync.WaitGroup
	for attempt := 0; attempt < 20; attempt++ {
		requests.Add(1)
		go func() { defer requests.Done(); manager.Apply(context.Background(), 42, "model-a", headers.Clone()) }()
	}
	requests.Wait()
	<-started
	close(proceed)
	require.Eventually(test, func() bool { return client.Exists(context.Background(), recordKey(42, "model-a")).Val() == 1 }, 2*time.Second, 10*time.Millisecond)
	require.EqualValues(test, 1, calls.Load())
	require.Eventually(test, func() bool { status := manager.Snapshot(42); return len(status) == 1 && status[0].State == "ready" }, time.Second, 10*time.Millisecond)
}

func TestDistributedLockPreventsParallelSampling(test *testing.T) {
	manager, client := managerForTest(test)
	require.NoError(test, client.Set(context.Background(), recordKey(42, "model-a")+":lock", "another-instance", time.Minute).Err())
	var calls atomic.Int64
	manager.sample = func(context.Context, http.Header, string) (Record, error) {
		calls.Add(1)
		return Record{}, errors.New("unexpected")
	}
	manager.Apply(context.Background(), 42, "model-a", make(http.Header))
	require.Eventually(test, func() bool {
		status := manager.Snapshot(42)
		return len(status) == 1 && status[0].LastError == "another instance is refreshing"
	}, time.Second, 10*time.Millisecond)
	require.Zero(test, calls.Load())
	require.Equal(test, "another-instance", client.Get(context.Background(), recordKey(42, "model-a")+":lock").Val())
}

func TestRefreshFailureKeepsExistingValidState(test *testing.T) {
	manager, client := managerForTest(test)
	value := stateValue(time.Now().Add(-56*time.Minute), 217)
	storeRecord(test, client, 42, "model-a", value)
	manager.config.Attempts = 1
	manager.sample = func(context.Context, http.Header, string) (Record, error) {
		return Record{}, errors.New("candidate length 312 is not accepted")
	}
	headers := make(http.Header)
	require.True(test, manager.Apply(context.Background(), 42, "model-a", headers))
	require.Equal(test, value, headers.Get(Header))
	require.Eventually(test, func() bool { status := manager.Snapshot(42); return len(status) == 1 && status[0].LastError != "" }, 3*time.Second, 10*time.Millisecond)
	record, err := manager.read(context.Background(), 42, "model-a")
	require.NoError(test, err)
	require.Equal(test, value, record.Value)
}

func TestDisableDuringAcquisitionDoesNotPublish(test *testing.T) {
	manager, client := managerForTest(test)
	var checks atomic.Int64
	manager.prepare = func(_ context.Context, _ int64, headers http.Header) (http.Header, bool) {
		return headers, checks.Add(1) == 1
	}
	manager.sample = func(_ context.Context, _ http.Header, model string) (Record, error) {
		return Parse(stateValue(time.Now(), 217), model, time.Now())
	}
	manager.Apply(context.Background(), 42, "model-a", make(http.Header))
	require.Eventually(test, func() bool {
		status := manager.Snapshot(42)
		return len(status) == 1 && status[0].LastError == "account disabled during acquisition"
	}, time.Second, 10*time.Millisecond)
	require.Zero(test, client.Exists(context.Background(), recordKey(42, "model-a")).Val())
}

func TestSampleHeadersDoNotCarryOldStateOrClientPayloadMetadata(test *testing.T) {
	source := make(http.Header)
	for _, name := range []string{Header, "Cookie", "Session-Id", "Conversation_Id", "X-Codex-Routing-Hint", "Proxy-Authorization"} {
		source.Set(name, "old")
	}
	source.Set("Authorization", "Bearer current")
	source.Set("Chatgpt-Account-Id", "account")
	result := sampleHeaders(source)
	require.Len(test, result, 2)
	require.Equal(test, "Bearer current", result.Get("Authorization"))
	require.Equal(test, "account", result.Get("Chatgpt-Account-Id"))
}

func TestConfigurationFailsClosed(test *testing.T) {
	manager, err := New(Config{}, nil, nil)
	require.NoError(test, err)
	require.Nil(test, manager)
	for _, endpoint := range []string{"http://pool.example", "https://user:secret@pool.example", "https://pool.example/path", "https://pool.example?token=secret"} {
		manager, err = New(Config{URL: endpoint, Token: strings.Repeat("x", 48)}, nil, nil)
		require.Error(test, err)
		require.Nil(test, manager)
	}
}
