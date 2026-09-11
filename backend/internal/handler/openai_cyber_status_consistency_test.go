package handler

import (
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/stretchr/testify/require"
)

func TestCyberPolicyOpsStatusConsistency(t *testing.T) {
	for _, tc := range []struct{ client, upstream int }{{403, 403}, {502, 403}, {200, 400}, {101, 200}} {
		entry := buildCyberPolicyOpsErrorEntry(cyberPolicyOpsErrorMeta{ClientStatusCode: tc.client}, &service.CyberPolicyMark{UpstreamStatus: tc.upstream})
		require.Equal(t, tc.client, entry.StatusCode)
		require.NotNil(t, entry.UpstreamStatusCode)
		require.Equal(t, tc.upstream, *entry.UpstreamStatusCode)
	}
}
