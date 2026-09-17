package turnstate

import (
	"context"
	"encoding/pem"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestRuntimeUpdateKeepsCachedStateAndActiveTargets(test *testing.T) {
	original, cache := managerForTest(test)
	runtime := NewRuntime(func(config Config) (*Manager, error) { return New(config, cache, nil) })
	test.Cleanup(runtime.Close)
	require.NoError(test, runtime.Update(original.config, nil))
	value := stateValue(time.Now(), 249)
	storeRecord(test, cache, 42, "model", value)
	require.True(test, runtime.Apply(context.Background(), 42, "model", make(http.Header)))
	previous := runtime.manager
	config := original.config
	config.Countries = []string{"US", "DE"}
	require.NoError(test, runtime.Update(config, func() error { return nil }))
	require.True(test, previous.closed)
	require.Len(test, runtime.manager.Snapshot(42), 1)
	require.Equal(test, []string{"US", "DE"}, runtime.Countries())
	headers := make(http.Header)
	require.True(test, runtime.Apply(context.Background(), 42, "model", headers))
	require.Equal(test, value, headers.Get(Header))
	current := runtime.manager
	require.Error(test, runtime.Update(config, func() error { return errors.New("database unavailable") }))
	require.Same(test, current, runtime.manager)
	config.ProxyHost = "bad-address"
	require.Error(test, runtime.Update(config, nil))
	require.Same(test, current, runtime.manager)
}

func TestRuntimeConcurrentReadsAndUpdates(test *testing.T) {
	original, cache := managerForTest(test)
	runtime := NewRuntime(func(config Config) (*Manager, error) { return New(config, cache, nil) })
	test.Cleanup(runtime.Close)
	require.NoError(test, runtime.Update(original.config, nil))
	storeRecord(test, cache, 42, "model", stateValue(time.Now(), 249))
	var workers sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for attempt := 0; attempt < 25; attempt++ {
				runtime.Apply(context.Background(), 42, "model", make(http.Header))
				runtime.Configured(SourcePurchased)
				runtime.Countries()
				runtime.Inspect(context.Background(), 42, Options{})
			}
		}()
	}
	for attempt := 0; attempt < 5; attempt++ {
		require.NoError(test, runtime.Update(original.config, nil))
	}
	workers.Wait()
	require.NoError(test, runtime.Update(Config{}, nil))
	require.False(test, runtime.Configured(SourcePurchased))
	runtime.Close()
	require.Error(test, runtime.Update(original.config, nil))
}

func TestRuntimeAcceptsUploadedPoolCertificate(test *testing.T) {
	original, cache := managerForTest(test)
	server := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer server.Close()
	config := original.config
	config.CAPEM = string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: server.Certificate().Raw}))
	manager, err := New(config, cache, nil)
	require.NoError(test, err)
	manager.Close()
	config.CAPEM = "invalid certificate"
	_, err = New(config, cache, nil)
	require.ErrorContains(test, err, "certificate")
}

func TestRuntimeUpdateDoesNotBlockCachedBusinessRequests(test *testing.T) {
	original, cache := managerForTest(test)
	runtime := NewRuntime(func(config Config) (*Manager, error) { return New(config, cache, nil) })
	test.Cleanup(runtime.Close)
	require.NoError(test, runtime.Update(original.config, nil))
	storeRecord(test, cache, 42, "model", stateValue(time.Now(), 249))
	previous := runtime.manager
	previous.workers.Add(1)
	var release sync.Once
	defer release.Do(previous.workers.Done)
	updated := make(chan error, 1)
	go func() { updated <- runtime.Update(original.config, nil) }()
	require.Eventually(test, func() bool {
		previous.mu.Lock()
		defer previous.mu.Unlock()
		return previous.closed
	}, time.Second, time.Millisecond)
	applied := make(chan bool, 1)
	go func() { applied <- runtime.Apply(context.Background(), 42, "model", make(http.Header)) }()
	select {
	case success := <-applied:
		require.True(test, success)
	case <-time.After(time.Second):
		test.Fatal("configuration replacement blocked cached requests")
	}
	release.Do(previous.workers.Done)
	require.NoError(test, <-updated)
}
