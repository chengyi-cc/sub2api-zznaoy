package basispoints

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/stretchr/testify/require"
	"io"
	"strings"
	"sync"
	"testing"
)

type sharedFixtureEntry struct {
	raw     []byte
	version uint64
}
type sharedFixtureStore struct {
	mu                  sync.Mutex
	entries             map[string]sharedFixtureEntry
	failRead, failWrite bool
	revision            uint64
}

func (s *sharedFixtureStore) LoadBPSState(_ context.Context, scope, kind string) ([]byte, uint64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.failRead {
		return nil, 0, errors.New("synthetic storage failure")
	}
	v := s.entries[scope+"/"+kind]
	return append([]byte(nil), v.raw...), v.version, nil
}
func (s *sharedFixtureStore) SaveBPSState(_ context.Context, scope, kind string, expected *uint64, raw []byte) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.failWrite {
		return false, errors.New("synthetic storage failure")
	}
	key := scope + "/" + kind
	if expected != nil && s.entries[key].version != *expected {
		return false, nil
	}
	if s.entries == nil {
		s.entries = map[string]sharedFixtureEntry{}
	}
	s.revision++
	s.entries[key] = sharedFixtureEntry{append([]byte(nil), raw...), s.revision}
	return true, nil
}
func TestPersistentBridgeCrossInstanceAndFailureBeforeToolDelivery(t *testing.T) {
	store := &sharedFixtureStore{}
	replay, catalog := NewPersistentCaches(context.Background(), store)
	raw := catalogCacheBody(t, []any{catalogEcho("echo")}, true)
	_, bridge, err := PrepareWithCatalog(raw, "account:1/key:1/thread:x", replay, catalog)
	require.NoError(t, err)
	native := nativeCall(object{"name": "echo", "arguments": object{"text": "synthetic"}})
	call, err := bridge.translateCall(native)
	require.NoError(t, err)
	replay2, catalog2 := NewPersistentCaches(context.Background(), store)
	source := testSource()
	source["input"] = []any{object{"type": "function_call_output", "call_id": call["call_id"], "output": "ok"}}
	input, _ := json.Marshal(source)
	_, restored, err := PrepareWithCatalog(input, "account:1/key:1/thread:x", replay2, catalog2)
	require.NoError(t, err)
	require.Contains(t, restored.tools, "echo")
	_, _, err = PrepareWithCatalog(input, "account:2/key:1/thread:x", replay2, catalog2)
	require.Error(t, err)
	store.failWrite = true
	body := bridge.Stream(io.NopCloser(strings.NewReader(sse(object{"type": "response.completed", "response": object{"id": "resp_test", "output": []any{native}}}))))
	output, err := io.ReadAll(body)
	require.NoError(t, err)
	require.NoError(t, body.Close())
	require.Contains(t, string(output), "response.failed")
	require.NotContains(t, string(output), "response.function_call_arguments.delta")
	store.failWrite = false
	store.failRead = true
	replay3, catalog3 := NewPersistentCaches(context.Background(), store)
	_, _, err = PrepareWithCatalog(raw, "account:1/key:1/thread:x", replay3, catalog3)
	require.ErrorIs(t, err, ErrStateUnavailable)
}
