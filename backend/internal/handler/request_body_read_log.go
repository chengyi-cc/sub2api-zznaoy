package handler

import (
	"net/http"

	"github.com/Wei-Shaw/sub2api/internal/pkg/errorarchive"
	pkghttputil "github.com/Wei-Shaw/sub2api/internal/pkg/httputil"
	"go.uber.org/zap"
)

// requestBodyReadFailure maps a request body read error to the response status
// and message. Read timeouts (408) are no longer reported as malformed
// requests; other read failures keep the stable generic 400 message. Failures
// other than the configured size limit are logged with bounded, payload-free
// diagnostics (kind, declared length, bytes received and read duration) so
// operators can tell a stalled or slow upload from a broken client without
// logging request content.
func requestBodyReadFailure(reqLog *zap.Logger, req *http.Request, err error) (int, string) {
	failure := pkghttputil.DescribeRequestBodyReadFailure(req, err)
	if failure.Kind != pkghttputil.RequestBodyReadErrorMaxBytes {
		logRequestBodyReadFailure(reqLog, failure)
	}
	return failure.Status, requestBodyReadClientMessage(failure)
}

// requestBodyReadClientMessage keeps upstream's limit-naming 413 message and
// uses the errorarchive wording for every other read failure kind.
func requestBodyReadClientMessage(failure pkghttputil.RequestBodyReadFailure) string {
	if failure.Kind == pkghttputil.RequestBodyReadErrorMaxBytes {
		return failure.ClientMessage()
	}
	return errorarchive.ReadErrorMessageForKind(failure.Kind)
}

func logRequestBodyReadFailure(reqLog *zap.Logger, failure pkghttputil.RequestBodyReadFailure) {
	if reqLog != nil {
		reqLog.Warn("read request body failed", failure.LogFields()...)
	}
}
