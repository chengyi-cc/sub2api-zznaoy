package admin

import (
	"strconv"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/pkg/response"
	middleware2 "github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"

	"github.com/gin-gonic/gin"
)

// SiteReconHandler 站点对账（管理员专用）。
// 安全约束：上游访问令牌、Bark 地址只写不读；接口只回显尾号 / 主机名。
// 路由未登记进 adminaccess 权限目录，因此带细分权限的管理员 API Key 无法访问。
type SiteReconHandler struct {
	svc *service.SiteReconService
}

func NewSiteReconHandler(svc *service.SiteReconService) *SiteReconHandler {
	return &SiteReconHandler{svc: svc}
}

// Overview GET /api/v1/admin/site-recon/overview?day=
func (h *SiteReconHandler) Overview(c *gin.Context) {
	out, err := h.svc.Overview(c.Request.Context(), strings.TrimSpace(c.Query("day")))
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, out)
}

// Detail GET /api/v1/admin/site-recon/detail?day=
func (h *SiteReconHandler) Detail(c *gin.Context) {
	out, err := h.svc.Detail(c.Request.Context(), strings.TrimSpace(c.Query("day")))
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, out)
}

// Anomalies GET /api/v1/admin/site-recon/anomalies?day=
func (h *SiteReconHandler) Anomalies(c *gin.Context) {
	out, err := h.svc.Anomalies(c.Request.Context(), strings.TrimSpace(c.Query("day")))
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, out)
}

// Runs GET /api/v1/admin/site-recon/runs
func (h *SiteReconHandler) Runs(c *gin.Context) {
	out, err := h.svc.Runs(c.Request.Context())
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, out)
}

// GetRun GET /api/v1/admin/site-recon/runs/:id
func (h *SiteReconHandler) GetRun(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil || id <= 0 {
		response.BadRequest(c, "Invalid id")
		return
	}
	out, err := h.svc.GetRun(c.Request.Context(), id)
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, out)
}

// StartRun POST /api/v1/admin/site-recon/runs
func (h *SiteReconHandler) StartRun(c *gin.Context) {
	var body service.SiteReconStartInput
	if err := c.ShouldBindJSON(&body); err != nil {
		response.BadRequest(c, "Invalid request: "+err.Error())
		return
	}
	id, err := h.svc.StartManualRun(c.Request.Context(), body)
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, gin.H{"id": id})
}

// GetConfig GET /api/v1/admin/site-recon/config
func (h *SiteReconHandler) GetConfig(c *gin.Context) {
	out, err := h.svc.GetConfig(c.Request.Context())
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, out)
}

// UpdateConfig PUT /api/v1/admin/site-recon/config
func (h *SiteReconHandler) UpdateConfig(c *gin.Context) {
	var body service.SiteReconConfigUpdate
	if err := c.ShouldBindJSON(&body); err != nil {
		response.BadRequest(c, "Invalid request: "+err.Error())
		return
	}
	out, err := h.svc.UpdateConfig(c.Request.Context(), body)
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, out)
}

// TestPush POST /api/v1/admin/site-recon/config/test-push
func (h *SiteReconHandler) TestPush(c *gin.Context) {
	if err := h.svc.TestPush(c.Request.Context()); err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, gin.H{"ok": true})
}

// ListTokens GET /api/v1/admin/site-recon/tokens
func (h *SiteReconHandler) ListTokens(c *gin.Context) {
	out, err := h.svc.ListTokens(c.Request.Context())
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, out)
}

type siteReconTokenPayload struct {
	Host        string `json:"host"`
	AccessToken string `json:"access_token"`
}

// SetToken PUT /api/v1/admin/site-recon/tokens
func (h *SiteReconHandler) SetToken(c *gin.Context) {
	subject, ok := middleware2.GetAuthSubjectFromContext(c)
	if !ok {
		response.Unauthorized(c, "Not authenticated")
		return
	}
	var body siteReconTokenPayload
	if err := c.ShouldBindJSON(&body); err != nil {
		response.BadRequest(c, "Invalid request")
		return
	}
	middleware2.SetAuditExtra(c, map[string]any{"site_recon_host": strings.TrimSpace(body.Host)})
	if err := h.svc.SetToken(c.Request.Context(), body.Host, body.AccessToken, subject.UserID); err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, gin.H{"ok": true})
}

type siteReconTokenDeletePayload struct {
	Host string `json:"host"`
}

// DeleteToken POST /api/v1/admin/site-recon/tokens/delete
func (h *SiteReconHandler) DeleteToken(c *gin.Context) {
	var body siteReconTokenDeletePayload
	if err := c.ShouldBindJSON(&body); err != nil {
		response.BadRequest(c, "Invalid request")
		return
	}
	middleware2.SetAuditExtra(c, map[string]any{"site_recon_host": strings.TrimSpace(body.Host)})
	if err := h.svc.DeleteToken(c.Request.Context(), body.Host); err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, gin.H{"ok": true})
}
