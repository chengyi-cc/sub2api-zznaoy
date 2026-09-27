package basispoints

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCompatibleDuplicateToolAnnotationsAndSchemaAliases(t *testing.T) {
	for _, alias := range []string{"parameters", "inputSchema", "input_schema"} {
		t.Run(alias, func(t *testing.T) {
			schema := object{"type": "object", "required": []any{"cmd"}, "properties": object{"cmd": object{"type": "string"}}}
			source := testSource()
			source["tools"] = []any{object{"type": "function", "name": "shell", "description": "current description", "parameters": schema, "strict": true}}
			source["input"] = []any{object{"type": "additional_tools", "tools": []any{object{"type": "function", "name": "shell", "description": "older annotation", alias: schema, "strict": true, "defer_loading": true}}}, message("user", "continue")}
			_, b := mustPrepare(t, source, "scope", nil)
			require.Len(t, b.tools, 1)
			require.Equal(t, "current description", b.tools["shell"].Catalog["description"])
			require.Equal(t, schema, b.tools["shell"].Parameters)
		})
	}
	source := testSource()
	source["tools"] = []any{object{"type": "custom", "name": "apply_patch", "description": "current"}, object{"type": "custom", "name": "apply_patch", "description": "historical", "defer_loading": true}}
	_, b := mustPrepare(t, source, "scope", nil)
	require.Len(t, b.tools, 1)
}

func TestCompatibleDuplicateToolsRetainExecutableConstraints(t *testing.T) {
	for _, field := range []string{"type", "format", "strict", "parameters", "encrypted", "new_constraint"} {
		t.Run(field, func(t *testing.T) {
			source := testSource()
			first := object{"type": "custom", "name": "apply_patch", "description": "current"}
			second := object{"type": "custom", "name": "apply_patch", "description": "other"}
			second[field] = true
			if field == "type" {
				second[field] = "function"
			}
			source["tools"] = []any{first, second}
			raw, err := json.Marshal(source)
			require.NoError(t, err)
			_, _, err = Prepare(raw, "scope", nil)
			require.ErrorContains(t, err, "conflicting duplicate")
		})
	}
	for _, namespaced := range []bool{false, true} {
		source := testSource()
		first := object{"type": "custom", "name": "exec", "description": "declare const tools: { allowed(args: object): Promise<unknown>; };"}
		second := object{"type": "custom", "name": "exec", "description": "declare const tools: { revoked(args: object): Promise<unknown>; };"}
		var tools []any = []any{first, second}
		if namespaced {
			tools = []any{object{"type": "namespace", "name": "functions", "tools": tools}}
		}
		source["tools"] = tools
		raw, err := json.Marshal(source)
		require.NoError(t, err)
		_, _, err = Prepare(raw, "scope", nil)
		require.ErrorContains(t, err, "conflicting duplicate", "exec descriptions carry executable contracts, not just annotations")
	}
}
