package basispoints

import (
	"encoding/json"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"testing"
)

func TestCompactionOptionsDefaultDisableOverrideAndTrigger(t *testing.T) {
	source := testSource()
	source["input"] = []any{message("user", "hello"), object{"type": "compaction_trigger"}}
	raw, err := json.Marshal(source)
	require.NoError(t, err)
	legacy, _, err := Prepare(raw, "", nil)
	require.NoError(t, err)
	defaulted, _, err := PrepareWithOptions(raw, "", nil, PrepareOptions{})
	require.NoError(t, err)
	require.Equal(t, legacy, defaulted)
	require.EqualValues(t, 920000, gjson.GetBytes(defaulted, "context_management.0.compact_threshold").Int())
	for _, threshold := range []int{0, 320000} {
		body, _, err := PrepareWithOptions(raw, "", nil, PrepareOptions{CompactionThresholdTokens: &threshold})
		require.NoError(t, err)
		if threshold == 0 {
			require.Equal(t, "[]", gjson.GetBytes(body, "context_management").Raw)
		} else {
			require.EqualValues(t, threshold, gjson.GetBytes(body, "context_management.0.compact_threshold").Int())
		}
		input := gjson.GetBytes(body, "input").Array()
		require.Equal(t, "compaction_trigger", input[len(input)-1].Get("type").String())
	}
	for _, explicit := range []any{[]any{}, []any{object{"type": "compaction", "compact_threshold": 123456}}} {
		source["context_management"] = explicit
		raw, err = json.Marshal(source)
		require.NoError(t, err)
		threshold := 0
		body, _, err := PrepareWithOptions(raw, "", nil, PrepareOptions{CompactionThresholdTokens: &threshold})
		require.NoError(t, err)
		want, err := json.Marshal(explicit)
		require.NoError(t, err)
		require.JSONEq(t, string(want), gjson.GetBytes(body, "context_management").Raw)
	}
	invalid := -1
	_, _, err = PrepareWithOptions(raw, "", nil, PrepareOptions{CompactionThresholdTokens: &invalid})
	require.ErrorContains(t, err, "non-negative")
}

func TestCatalogCompactionPolicyIsPerRequest(t *testing.T) {
	cache := new(CatalogCache)
	source := testSource()
	source["tools"] = []any{object{"type": "function", "name": "shell", "parameters": object{"type": "object"}}}
	raw, err := json.Marshal(source)
	require.NoError(t, err)
	zero := 0
	body, _, err := PrepareWithCatalogOptions(raw, "scope", nil, cache, "catalog", PrepareOptions{CompactionThresholdTokens: &zero})
	require.NoError(t, err)
	require.Equal(t, "[]", gjson.GetBytes(body, "context_management").Raw)
	delete(source, "tools")
	raw, err = json.Marshal(source)
	require.NoError(t, err)
	body, bridge, err := PrepareWithCatalogOptions(raw, "scope", nil, cache, "catalog", PrepareOptions{})
	require.NoError(t, err)
	require.Len(t, bridge.tools, 1)
	require.EqualValues(t, 920000, gjson.GetBytes(body, "context_management.0.compact_threshold").Int())
	source["tool_choice"] = "none"
	raw, err = json.Marshal(source)
	require.NoError(t, err)
	body, _, err = PrepareWithCatalogOptions(raw, "scope", nil, cache, "catalog", PrepareOptions{CompactionThresholdTokens: &zero})
	require.NoError(t, err)
	require.Equal(t, "[]", gjson.GetBytes(body, "context_management").Raw)
}
