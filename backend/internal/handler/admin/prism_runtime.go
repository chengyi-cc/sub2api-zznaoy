package admin

import (
	"github.com/Wei-Shaw/sub2api/internal/pkg/response"
	"github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/gin-gonic/gin"
	"net/http"
)

func (h *AccountHandler) GetPrismRuntime(c *gin.Context)     { h.prismRuntime(c, "status") }
func (h *AccountHandler) GetPrismRuntimeLogs(c *gin.Context) { h.prismRuntime(c, "logs") }
func (h *AccountHandler) ControlPrismRuntime(c *gin.Context) {
	action := c.Param("action")
	switch action {
	case "start", "stop", "restart", "check":
		middleware.SetAuditAction(c, "admin.prism."+action)
		h.prismRuntime(c, action)
	default:
		response.BadRequest(c, "Invalid Prism service action")
	}
}
func (h *AccountHandler) prismRuntime(c *gin.Context, action string) {
	c.Header("Cache-Control", "no-store")
	status, err := h.accountTestService.PrismRuntime(c.Request.Context(), action)
	if err != nil {
		response.Error(c, http.StatusServiceUnavailable, err.Error())
		return
	}
	response.Success(c, status)
}
