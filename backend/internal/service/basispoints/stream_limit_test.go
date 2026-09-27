package basispoints

import (
	"errors"
	"io"
	"strings"
	"testing"
	"testing/iotest"

	"github.com/stretchr/testify/require"
)

func TestSSEConfiguredLimitBoundsLinesAndMultilineEvents(t *testing.T) {
	for _, tc := range []struct {
		name, wire, want string
		limit            int
		failure          string
	}{
		{"single line", "event: sample\ndata: abc\n\n", "abc", 24, ""},
		{"multiline", "data: abc\ndata: def\n\n", "abc\ndef", 16, ""},
		{"line bound", "data: " + strings.Repeat("x", 20) + "\n\n", "", 24, "line exceeds 24 bytes"},
		{"aggregate bound", strings.Repeat("data: 1234567\n", 4) + "\n", "", 24, "event exceeds 24 bytes"},
		{"aggregate exact bound", strings.Repeat("data: 1234567\n", 3) + "\n", "1234567\n1234567\n1234567", 24, ""},
		{"nonpositive fallback", "data: abc\n\n", "abc", 0, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, fragmented := range []bool{false, true} {
				var r io.Reader = strings.NewReader(tc.wire)
				if fragmented {
					r = iotest.OneByteReader(r)
				}
				var got []string
				err := readEventsWithLimit(r, tc.limit, func(_ string, data []byte) error {
					got = append(got, string(data))
					return nil
				})
				if tc.failure != "" {
					require.ErrorContains(t, err, tc.failure)
					var invalid protocolError
					require.True(t, errors.As(err, &invalid))
					require.Empty(t, got, "oversized event must not be partially dispatched")
				} else {
					require.NoError(t, err)
					require.Equal(t, []string{tc.want}, got)
				}
			}
		})
	}
}
