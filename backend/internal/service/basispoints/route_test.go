package basispoints

import "testing"

func TestNativeCapabilityRouting(t *testing.T) {
	for _, tc := range []struct {
		body   string
		native bool
	}{
		{`{"tools":[{"type":"web_search"}]}`, true},
		{`{"tools":[{"type":"image_generation"}]}`, true},
		{`{"tools":[{"type":"namespace","name":"nested","tools":[{"type":"file_search"}]}]}`, true},
		{`{"input":[{"type":"additional_tools","tools":[{"type":"tool_search"}]}]}`, true},
		{`{"tool_choice":"required"}`, true},
		{`{"tool_choice":{"type":"function","name":"foo"}}`, true},
		{`{"input":[{"role":"user","content":[{"type":"input_file","file_id":"file-test"}]}]}`, true},
		{`{"tools":[{"type":"function","name":"image_generation"}],"tool_choice":"auto"}`, false},
		{`{"tools":[{"type":"custom","name":"exec"}],"tool_choice":"none"}`, false},
		{`{"input":[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,x"}]}]}`, false},
	} {
		if got := NativeFallbackReason([]byte(tc.body)); (got != "") != tc.native {
			t.Errorf("%s: %q", tc.body, got)
		}
	}
}
