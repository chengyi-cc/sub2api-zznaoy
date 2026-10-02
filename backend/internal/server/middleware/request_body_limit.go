package middleware

import (
	"github.com/Wei-Shaw/sub2api/internal/pkg/httputil"
	"net/http"

	"github.com/gin-gonic/gin"
)

// RequestBodyLimit 使用 MaxBytesReader 限制请求体大小。
func RequestBodyLimit(maxBytes int64) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Request = c.Request.WithContext(httputil.WithRequestBodyLimit(c.Request.Context(), maxBytes))
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxBytes)
		c.Next()
	}
}
