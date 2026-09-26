package handler

import (
	"github.com/Wei-Shaw/sub2api/internal/pkg/errorarchive"
	"net/http"
	"strings"

	"go.uber.org/zap"
)

// logRequestBodyReadFailure records a bounded, payload-free reason for a body
// read failure. Operators get enough information to distinguish compression failures from a
// disconnected/truncated upload without logging request content.
func logRequestBodyReadFailure(reqLog *zap.Logger, req *http.Request, err error) {
	if reqLog == nil || err == nil {
		return
	}

	contentLength := int64(-1)
	contentEncoding := "identity"
	if req != nil {
		contentLength = req.ContentLength
		contentEncoding = requestContentEncodingCategory(req.Header.Get("Content-Encoding"))
	}

	reqLog.Warn("read request body failed",
		zap.String("error_kind", requestBodyReadErrorKind(err)),
		zap.String("content_encoding", contentEncoding),
		zap.Int64("content_length", contentLength),
	)
}

func requestContentEncodingCategory(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "identity":
		return "identity"
	case "gzip", "x-gzip":
		return "gzip"
	case "zstd":
		return "zstd"
	case "deflate":
		return "deflate"
	default:
		return "other"
	}
}

func requestBodyReadErrorKind(err error) string {
	if err == nil {
		return "none"
	}
	return errorarchive.ReadErrorKind(err)
}
