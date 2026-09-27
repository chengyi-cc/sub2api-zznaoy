package basispoints

import (
	"encoding/json"
	"fmt"
	"github.com/stretchr/testify/require"
	"strings"
	"sync"
	"testing"
	"time"
)

func catalogCacheBody(t *testing.T, tools any, explicit bool, additions ...any) []byte {
	t.Helper()
	request := object{"model": "gpt-5.6-sol", "input": []any{message("user", "hello")}}
	if explicit {
		request["tools"] = tools
	}
	if len(additions) > 0 {
		request["input"] = append(request["input"].([]any), object{"type": "additional_tools", "tools": additions})
	}
	raw, err := json.Marshal(request)
	require.NoError(t, err)
	return raw
}
func catalogEcho(name string) object {
	return object{"type": "function", "name": name, "parameters": object{"type": "object"}}
}

func TestCatalogCacheExplicitRequestsPreservePreparedBytesAndReplayScope(t *testing.T) {
	raw := catalogCacheBody(t, []any{catalogEcho("z"), catalogEcho("a")}, true)
	baseline, _, err := Prepare(raw, "old-replay-scope", nil)
	require.NoError(t, err)
	cache := new(CatalogCache)
	for i := 0; i < 3; i++ {
		got, b, err := PrepareWithCatalog(raw, "old-replay-scope", nil, cache, "isolated-owner")
		require.NoError(t, err)
		require.Equal(t, baseline, got)
		require.Equal(t, "old-replay-scope", b.scope)
	}
}
func TestCatalogCacheInheritMergeReplaceAndRevoke(t *testing.T) {
	cache := new(CatalogCache)
	tools := []any{object{"type": "namespace", "name": "client", "tools": []any{catalogEcho("echo")}}}
	_, _, err := PrepareWithCatalog(catalogCacheBody(t, tools, true), "s", nil, cache)
	require.NoError(t, err)
	omitted := catalogCacheBody(t, nil, false)
	_, b, err := PrepareWithCatalog(omitted, "s", nil, cache)
	require.NoError(t, err)
	require.Equal(t, "client", b.tools["client.echo"].Namespace)
	require.Equal(t, "echo", b.tools["client.echo"].Name)
	first, _, err := PrepareWithCatalog(omitted, "s", nil, cache)
	require.NoError(t, err)
	second, _, err := PrepareWithCatalog(omitted, "s", nil, cache)
	require.NoError(t, err)
	require.Equal(t, first, second)
	_, b, err = PrepareWithCatalog(catalogCacheBody(t, nil, false, catalogEcho("extra")), "s", nil, cache)
	require.NoError(t, err)
	require.Len(t, b.tools, 2)
	_, b, err = PrepareWithCatalog(catalogCacheBody(t, []any{catalogEcho("replacement")}, true), "s", nil, cache)
	require.NoError(t, err)
	require.Len(t, b.tools, 1)
	require.Contains(t, b.tools, "replacement")
	_, _, err = PrepareWithCatalog(catalogCacheBody(t, []any{}, true), "s", nil, cache)
	require.NoError(t, err)
	_, b, err = PrepareWithCatalog(omitted, "s", nil, cache)
	require.NoError(t, err)
	require.Empty(t, b.tools)
}
func TestCatalogCacheCannotCrossScopeOrPoisonValidatedState(t *testing.T) {
	cache := new(CatalogCache)
	raw := catalogCacheBody(t, []any{catalogEcho("echo")}, true)
	_, _, err := PrepareWithCatalog(raw, "s", nil, cache, "owner1/key1/thread1")
	require.NoError(t, err)
	for _, scope := range []string{"owner2/key1/thread1", "owner1/key2/thread1", "owner1/key1/thread2"} {
		_, b, err := PrepareWithCatalog(catalogCacheBody(t, nil, false), "s", nil, cache, scope)
		require.NoError(t, err)
		require.Empty(t, b.tools)
	}
	invalid := catalogCacheBody(t, []any{catalogEcho("echo"), object{"type": "custom", "name": "echo"}}, true)
	_, _, err = PrepareWithCatalog(invalid, "s", nil, cache, "owner1/key1/thread1")
	require.Error(t, err)
	_, b, err := PrepareWithCatalog(catalogCacheBody(t, nil, false), "s", nil, cache, "owner1/key1/thread1")
	require.NoError(t, err)
	require.Contains(t, b.tools, "echo")
	_, b, err = PrepareWithCatalog(catalogCacheBody(t, nil, false), "s", nil, nil, "owner1/key1/thread1")
	require.NoError(t, err)
	require.Empty(t, b.tools)
}
func TestCatalogCacheNoneDoesNotEraseToolsAndExpiryWorks(t *testing.T) {
	cache := new(CatalogCache)
	now := time.Now()
	cache.now = func() time.Time { return now }
	_, _, err := PrepareWithCatalog(catalogCacheBody(t, []any{catalogEcho("echo")}, true), "s", nil, cache)
	require.NoError(t, err)
	_, b, err := PrepareWithCatalog([]byte("{\"model\":\"gpt-5.6-sol\",\"input\":\"hi\",\"tool_choice\":\"none\",\"tools\":[]}"), "s", nil, cache)
	require.NoError(t, err)
	require.Empty(t, b.tools)
	_, b, err = PrepareWithCatalog(catalogCacheBody(t, nil, false), "s", nil, cache)
	require.NoError(t, err)
	require.Contains(t, b.tools, "echo")
	now = now.Add(catalogCacheIdleTTL + time.Second)
	_, b, err = PrepareWithCatalog(catalogCacheBody(t, nil, false), "s", nil, cache)
	require.NoError(t, err)
	require.Empty(t, b.tools)
}
func TestCatalogCacheOversizeDoesNotRejectOrReuseRevokedTools(t *testing.T) {
	cache := new(CatalogCache)
	_, _, err := PrepareWithCatalog(catalogCacheBody(t, []any{catalogEcho("old")}, true), "s", nil, cache)
	require.NoError(t, err)
	huge := catalogEcho("large")
	huge["description"] = strings.Repeat("x", (1<<20)+1)
	_, b, err := PrepareWithCatalog(catalogCacheBody(t, []any{huge}, true), "s", nil, cache)
	require.NoError(t, err)
	require.Contains(t, b.tools, "large")
	_, b, err = PrepareWithCatalog(catalogCacheBody(t, nil, false), "s", nil, cache)
	require.NoError(t, err)
	require.Empty(t, b.tools)
}
func TestCatalogCacheConcurrentReplaceAndVersionChecks(t *testing.T) {
	cache := new(CatalogCache)
	require.True(t, cache.commit("guard", 0, []byte("[]")))
	require.False(t, cache.commit("guard", 0, []byte("[1]")))
	var wg sync.WaitGroup
	for i := 0; i < 24; i++ {
		raw := catalogCacheBody(t, []any{catalogEcho(fmt.Sprintf("tool%d", i))}, true)
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _, err := PrepareWithCatalog(raw, "s", nil, cache)
			if err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	_, b, err := PrepareWithCatalog(catalogCacheBody(t, nil, false), "s", nil, cache)
	require.NoError(t, err)
	require.Len(t, b.tools, 1)
	for i := 0; i < 600; i++ {
		require.True(t, cache.commit(fmt.Sprint(i), 0, []byte("[]")))
	}
	require.LessOrEqual(t, len(cache.entries), 512)
	require.LessOrEqual(t, cache.bytes, 16<<20)
}
