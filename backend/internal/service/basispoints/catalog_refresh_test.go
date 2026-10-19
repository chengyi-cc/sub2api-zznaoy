package basispoints

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCatalogCacheCurrentDeclarationsRefreshInheritedTools(t *testing.T) {
	for _, namespace := range []string{"", "functions"} {
		t.Run(namespace, func(t *testing.T) {
			wrap := func(decl object) any {
				if namespace == "" {
					return decl
				}
				return object{"type": "namespace", "name": namespace, "tools": []any{decl}}
			}
			old := object{"type": "custom", "name": "exec", "description": "old runtime tool list"}
			current := object{"type": "custom", "name": "exec", "description": "updated runtime tool list"}
			cache := new(CatalogCache)
			_, _, err := PrepareWithCatalog(catalogCacheBody(t, []any{wrap(old)}, true), "s", nil, cache)
			require.NoError(t, err)
			raw := catalogCacheBody(t, nil, false, wrap(current))
			cold, _, err := Prepare(raw, "s", nil)
			require.NoError(t, err)
			for i := 0; i < 2; i++ {
				warm, _, err := PrepareWithCatalog(raw, "s", nil, cache)
				require.NoError(t, err)
				require.Equal(t, cold, warm, "cached metadata must not alter the current declared contract")
			}
			_, b, err := PrepareWithCatalog(catalogCacheBody(t, nil, false), "s", nil, cache)
			require.NoError(t, err)
			key := "exec"
			if namespace != "" {
				key = namespace + "." + key
			}
			require.Equal(t, current, b.tools[key].Catalog)
		})
	}
}

func TestCatalogCacheRefreshPreservesOtherToolsAndRejectsCurrentConflicts(t *testing.T) {
	cache := new(CatalogCache)
	old := catalogEcho("echo")
	_, _, err := PrepareWithCatalog(catalogCacheBody(t, []any{old, catalogEcho("keep")}, true), "s", nil, cache)
	require.NoError(t, err)
	current := catalogEcho("echo")
	current["parameters"] = object{"type": "object", "required": []any{"text"}, "properties": object{"text": object{"type": "string"}}}
	_, b, err := PrepareWithCatalog(catalogCacheBody(t, nil, false, current), "s", nil, cache)
	require.NoError(t, err)
	require.Len(t, b.tools, 2)
	require.Equal(t, current, b.tools["echo"].Catalog)
	require.Contains(t, b.tools, "keep")
	_, _, err = PrepareWithCatalog(catalogCacheBody(t, nil, false, old, current), "s", nil, cache)
	require.ErrorContains(t, err, "conflicting duplicate")
	_, _, err = PrepareWithCatalog(catalogCacheBody(t, []any{old}, true, current), "s", nil, cache)
	require.ErrorContains(t, err, "conflicting duplicate")
	conflictsAcrossItems := testSource()
	conflictsAcrossItems["input"] = []any{object{"type": "additional_tools", "tools": []any{old}}, object{"type": "additional_tools", "tools": []any{current}}}
	conflictRaw, err := json.Marshal(conflictsAcrossItems)
	require.NoError(t, err)
	_, _, err = PrepareWithCatalog(conflictRaw, "s", nil, cache)
	require.ErrorContains(t, err, "conflicting duplicate")
	_, b, err = PrepareWithCatalog(catalogCacheBody(t, nil, false), "s", nil, cache)
	require.NoError(t, err)
	require.Equal(t, current, b.tools["echo"].Catalog, "rejected requests must not poison validated state")
}

// Opt-in local replay: no conversation data or credentials enter the repository
// or leave this process. Normal regression tests use synthetic declarations.
func TestCatalogCacheCapturedRequestRefresh(t *testing.T) {
	path := os.Getenv("BPS_CATALOG_CAPTURE_PATH")
	if path == "" {
		t.Skip("set BPS_CATALOG_CAPTURE_PATH for a local, offline error archive replay")
	}
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	var capture struct {
		RequestWireBase64 string `json:"request_wire_base64"`
		UploadComplete    bool   `json:"upload_complete"`
		RequestTruncated  bool   `json:"request_truncated"`
	}
	require.NoError(t, json.Unmarshal(data, &capture))
	require.True(t, capture.UploadComplete)
	require.False(t, capture.RequestTruncated)
	raw, err := base64.StdEncoding.DecodeString(capture.RequestWireBase64)
	require.NoError(t, err)
	_, _, err = Prepare(raw, "offline-replay", nil)
	require.NoError(t, err, "the capture alone must be a valid current catalog")
	cache := new(CatalogCache)
	stale := object{"type": "custom", "name": "exec", "description": "stale tool runtime description"}
	_, _, err = PrepareWithCatalog(catalogCacheBody(t, []any{stale}, true), "offline-replay", nil, cache)
	require.NoError(t, err)
	for i := 0; i < 2; i++ {
		_, _, err = PrepareWithCatalog(raw, "offline-replay", nil, cache)
		require.NoError(t, err, "current declarations must supersede stale cached declarations")
	}
}
