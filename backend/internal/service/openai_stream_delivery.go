package service

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// flushOpenAIStream preserves Gin's header state while observing transport
// flush errors. ResponseController used directly on Gin would select its
// error-discarding Flush method before considering Unwrap.
func flushOpenAIStream(w gin.ResponseWriter) error {
	w.WriteHeaderNow()
	if f, ok := w.(interface{ FlushError() error }); ok {
		return f.FlushError()
	}
	if u, ok := w.(interface{ Unwrap() http.ResponseWriter }); ok {
		return http.NewResponseController(u.Unwrap()).Flush()
	}
	// Wrappers without an error-capable flush or explicit unwrap keep their own
	// behavior; do not bypass their buffering or compression implementation.
	w.Flush()
	return nil
}
