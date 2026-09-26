package errorarchive

import (
	"context"
	"errors"
	"fmt"
	"io"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestReadErrorPublicMessageNeverEchoesPayload(t *testing.T) {
	for _, tc := range []struct {
		err           error
		kind, message string
	}{
		{fmt.Errorf("private-request: %w", io.ErrUnexpectedEOF), "truncated_body", "Request body upload was incomplete; resend the complete request"},
		{context.Canceled, "client_disconnect", "Request body upload was interrupted; resend the complete request"},
		{context.DeadlineExceeded, "read_timeout", "Request body upload timed out; resend the complete request"},
		{errors.New("decode Content-Encoding: unsupported Content-Encoding private-request"), "unsupported_content_encoding", "Unsupported request Content-Encoding"},
		{errors.New("decode Content-Encoding: private-request"), "decode_content_encoding", "Failed to decode the compressed request body"},
		{errors.New("private-request"), "io_read", "Failed to read request body"},
	} {
		require.Equal(t, tc.kind, ReadErrorKind(tc.err))
		require.Equal(t, tc.message, ReadErrorMessage(tc.err))
		require.NotContains(t, ReadErrorMessage(tc.err), "private-request")
	}
}
