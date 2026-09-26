package admin

import (
	"github.com/Wei-Shaw/sub2api/internal/pkg/response"
	"github.com/gin-gonic/gin"
	"net/http"
)

// Both routes live exclusively under the existing administrator-authenticated
// Ops route group. Captures are never included in ordinary error list responses.
func (h *OpsHandler) GetErrorArchive(c *gin.Context) {
	if h.opsService == nil {
		response.Error(c, 503, "Ops service unavailable")
		return
	}
	entry, err := h.opsService.ReadErrorArchive(c.Request.Context(), c.Param("archive_id"))
	if err != nil {
		response.Error(c, http.StatusNotFound, "Diagnostic capture unavailable, expired, evicted, or not yet persisted")
		return
	}
	c.Header("Cache-Control", "no-store")
	c.Header("X-Content-Type-Options", "nosniff")
	c.Header("Content-Disposition", "attachment; filename=error-capture-"+entry.ID+".json")
	c.JSON(http.StatusOK, entry)
}

func (h *OpsHandler) GetErrorArchiveHealth(c *gin.Context) {
	if h.opsService == nil {
		response.Error(c, 503, "Ops service unavailable")
		return
	}
	response.Success(c, h.opsService.ErrorArchiveStats())
}
