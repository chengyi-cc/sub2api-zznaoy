package handler

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/pkg/ctxkey"
	"github.com/Wei-Shaw/sub2api/internal/pkg/errorarchive"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
)

const opsErrorArchiveRefKey = "ops_error_archive_ref"

func opsArchiveRefFromContext(c *gin.Context) errorarchive.Ref {
	value, _ := c.Get(opsErrorArchiveRefKey)
	ref, _ := value.(errorarchive.Ref)
	return ref
}

// A bounded tee observes the bytes already read by normal handlers. It never
// eagerly reads, retries an upload, or delays the request waiting for disk I/O.
func beginOpsErrorArchive(c *gin.Context, ops *service.OpsService) func(int, []byte, bool) errorarchive.Ref {
	if c.Request == nil || c.Request.Body == nil || c.Request.Method != http.MethodPost {
		return nil
	}
	path := c.Request.URL.Path
	if !strings.Contains(path, "responses") && !strings.Contains(path, "chat/completions") && !strings.HasSuffix(path, "/messages") {
		return nil
	}
	if typ := strings.ToLower(c.GetHeader("Content-Type")); typ != "" && !strings.Contains(typ, "json") {
		return nil
	}
	capture, enabled := ops.CaptureErrorRequest(c.Request.Context(), c.Request.Body)
	if !enabled {
		return nil
	}
	encoding := requestContentEncodingCategory(c.GetHeader("Content-Encoding"))
	length := c.Request.ContentLength
	var trace *errorarchive.Trace
	if capture != nil {
		c.Request.Body = capture
		ctx, t := errorarchive.WithTrace(c.Request.Context())
		trace = t
		c.Request = c.Request.WithContext(ctx)
	}
	return func(status int, response []byte, visibleFailure bool) errorarchive.Ref {
		if capture != nil {
			defer capture.Release()
		}
		key := getOpsAPIKey(c)
		// Never retain unauthenticated input, authentication failures, or healthy requests.
		if key == nil || status == 401 || status == 403 || !ops.IsMonitoringEnabled(c.Request.Context()) {
			return errorarchive.Ref{}
		}
		diagnostics := trace.Snapshot()
		if status < 400 && !visibleFailure && len(diagnostics) == 0 {
			return errorarchive.Ref{}
		}
		requestID, _ := c.Request.Context().Value(ctxkey.RequestID).(string)
		if requestID == "" {
			requestID = c.Writer.Header().Get("X-Request-ID")
		}
		responseTruncated := len(response) > 64<<10
		if responseTruncated {
			response = response[:64<<10]
		}
		accountID := c.GetInt64(opsAccountIDKey)
		entry := &errorarchive.Entry{RequestID: requestID, APIKeyID: key.ID, AccountID: accountID, Path: path, Status: status, ContentEncoding: encoding, ContentLength: length, Response: append([]byte(nil), response...), Diagnostics: diagnostics}
		entry.ResponseTruncated = responseTruncated
		return capture.Save(entry)
	}
}

func withOpsArchiveRef(body string, ref errorarchive.Ref) string {
	if ref.State == "" {
		return body
	}
	// Keep the archive reference intact through the 8 KiB persistence limit.
	// The bounded encrypted capture retains the original response separately.
	body, _ = service.SanitizeOpsErrorBodyForQueue(body)
	if len(body) > 1024 {
		body = truncateString(body, 1024)
	}
	var content any = body
	if json.Valid([]byte(body)) {
		content = json.RawMessage(body)
	}
	raw, err := json.Marshal(map[string]any{"client_response": content, "diagnostic_archive": ref})
	if err != nil {
		return body
	}
	return string(raw)
}
