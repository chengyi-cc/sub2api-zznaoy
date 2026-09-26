package errorarchive

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"strings"
	"syscall"
)

// OnBodyReadError keeps a fixed category, never an error string that can echo data.
func (c *Capture) OnBodyReadError(err error) { c.errKind = ReadErrorKind(err) }

func ReadErrorKind(err error) string {
	if err == nil {
		return ""
	}
	var maxErr *http.MaxBytesError
	if errors.As(err, &maxErr) {
		return "max_bytes"
	}
	lower := strings.ToLower(err.Error())
	if strings.Contains(lower, "decode content-encoding") {
		if strings.Contains(lower, "unsupported content-encoding") {
			return "unsupported_content_encoding"
		}
		return "decode_content_encoding"
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, syscall.ECONNRESET) || errors.Is(err, syscall.EPIPE) {
		return "client_disconnect"
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return "read_timeout"
	}
	if errors.Is(err, io.ErrUnexpectedEOF) {
		return "truncated_body"
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		if netErr.Timeout() {
			return "read_timeout"
		}
		return "transport"
	}
	return "io_read"
}

// Fixed public messages never echo error strings, paths, or request content.
func ReadErrorMessage(err error) string {
	switch ReadErrorKind(err) {
	case "truncated_body":
		return "Request body upload was incomplete; resend the complete request"
	case "client_disconnect":
		return "Request body upload was interrupted; resend the complete request"
	case "read_timeout":
		return "Request body upload timed out; resend the complete request"
	case "unsupported_content_encoding":
		return "Unsupported request Content-Encoding"
	case "decode_content_encoding":
		return "Failed to decode the compressed request body"
	default:
		return "Failed to read request body"
	}
}
