package admin

import (
	"database/sql"
	"errors"
	"net/http"
	"strconv"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/pkg/response"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
)

type CandyMonitorHandler struct{ svc *service.CandyMonitorService }

func NewCandyMonitorHandler(svc *service.CandyMonitorService) *CandyMonitorHandler {
	return &CandyMonitorHandler{svc: svc}
}
func candyMonitorError(c *gin.Context, err error) {
	switch {
	case errors.Is(err, service.ErrCandyMonitorBusy):
		response.Error(c, http.StatusConflict, err.Error())
	case errors.Is(err, service.ErrCandyMonitorInvalid):
		response.BadRequest(c, err.Error())
	case errors.Is(err, sql.ErrNoRows):
		response.NotFound(c, "Candy monitor record not found")
	default:
		response.ErrorFrom(c, err)
	}
}
func candyMonitorID(c *gin.Context) (int64, bool) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil || id <= 0 {
		response.BadRequest(c, "Invalid ID")
		return 0, false
	}
	return id, true
}
func (h *CandyMonitorHandler) Settings(c *gin.Context) {
	v, err := h.svc.Settings(c.Request.Context())
	if err != nil {
		candyMonitorError(c, err)
		return
	}
	response.Success(c, v)
}
func (h *CandyMonitorHandler) SaveSettings(c *gin.Context) {
	var v service.CandyMonitorSettings
	if err := c.ShouldBindJSON(&v); err != nil {
		response.BadRequest(c, "Invalid settings")
		return
	}
	if err := h.svc.SaveSettings(c.Request.Context(), &v); err != nil {
		candyMonitorError(c, err)
		return
	}
	response.Success(c, v)
}
func (h *CandyMonitorHandler) List(c *gin.Context) {
	page, _ := strconv.Atoi(c.DefaultQuery("page", "1"))
	size, _ := strconv.Atoi(c.DefaultQuery("page_size", "30"))
	group, groupErr := strconv.ParseInt(c.DefaultQuery("group_id", "0"), 10, 64)
	if groupErr != nil || group < 0 {
		response.BadRequest(c, "Invalid group ID")
		return
	}
	if page < 1 {
		page = 1
	}
	if size < 1 || size > 100 {
		size = 30
	}
	search := strings.TrimSpace(c.Query("search"))
	if len(search) > 200 {
		response.BadRequest(c, "Search is too long")
		return
	}
	items, total, err := h.svc.List(c.Request.Context(), service.CandyMonitorFilter{GroupID: group, Search: search, EnabledOnly: c.Query("enabled_only") == "true", Page: page, PageSize: size})
	if err != nil {
		candyMonitorError(c, err)
		return
	}
	response.Paginated(c, items, total, page, size)
}
func (h *CandyMonitorHandler) Configure(c *gin.Context) {
	var req struct {
		AccountIDs []int64 `json:"account_ids"`
		service.CandyMonitorConfig
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		response.BadRequest(c, "Invalid configuration")
		return
	}
	if err := h.svc.Configure(c.Request.Context(), req.AccountIDs, req.CandyMonitorConfig); err != nil {
		candyMonitorError(c, err)
		return
	}
	response.Success(c, gin.H{"updated": true})
}
func (h *CandyMonitorHandler) Run(c *gin.Context) {
	id, ok := candyMonitorID(c)
	if !ok {
		return
	}
	var req struct {
		ModelID string `json:"model_id"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		response.BadRequest(c, "Invalid request")
		return
	}
	v, err := h.svc.Queue(c.Request.Context(), id, req.ModelID)
	if err != nil {
		candyMonitorError(c, err)
		return
	}
	response.Accepted(c, v)
}
func (h *CandyMonitorHandler) SetEnabled(c *gin.Context) {
	var req struct {
		AccountIDs []int64 `json:"account_ids"`
		Enabled    *bool   `json:"enabled"`
	}
	if err := c.ShouldBindJSON(&req); err != nil || req.Enabled == nil {
		response.BadRequest(c, "Invalid request")
		return
	}
	if err := h.svc.SetEnabled(c.Request.Context(), req.AccountIDs, *req.Enabled); err != nil {
		candyMonitorError(c, err)
		return
	}
	response.Success(c, gin.H{"updated": true})
}
func (h *CandyMonitorHandler) Result(c *gin.Context) {
	id, ok := candyMonitorID(c)
	if !ok {
		return
	}
	v, err := h.svc.Result(c.Request.Context(), id)
	if err != nil {
		candyMonitorError(c, err)
		return
	}
	response.Success(c, v)
}
func (h *CandyMonitorHandler) History(c *gin.Context) {
	id, ok := candyMonitorID(c)
	if !ok {
		return
	}
	v, err := h.svc.History(c.Request.Context(), id)
	if err != nil {
		candyMonitorError(c, err)
		return
	}
	response.Success(c, v)
}
