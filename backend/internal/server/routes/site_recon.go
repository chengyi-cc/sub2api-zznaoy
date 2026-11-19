package routes

import (
	"github.com/Wei-Shaw/sub2api/internal/handler"

	"github.com/gin-gonic/gin"
)

// registerSiteReconAdminRoutes 站点对账（fork 自有模块）。只挂在管理员组下，
// 不登记进 adminaccess 权限目录：细分权限的管理员 API Key 无法访问。
func registerSiteReconAdminRoutes(admin *gin.RouterGroup, h *handler.Handlers) {
	if h == nil || h.Admin == nil || h.Admin.SiteRecon == nil {
		return
	}
	r := admin.Group("/site-recon")
	{
		r.GET("/overview", h.Admin.SiteRecon.Overview)
		r.GET("/detail", h.Admin.SiteRecon.Detail)
		r.GET("/anomalies", h.Admin.SiteRecon.Anomalies)
		r.GET("/runs", h.Admin.SiteRecon.Runs)
		r.GET("/runs/:id", h.Admin.SiteRecon.GetRun)
		r.POST("/runs", h.Admin.SiteRecon.StartRun)
		r.GET("/config", h.Admin.SiteRecon.GetConfig)
		r.PUT("/config", h.Admin.SiteRecon.UpdateConfig)
		r.POST("/config/test-push", h.Admin.SiteRecon.TestPush)
		r.GET("/tokens", h.Admin.SiteRecon.ListTokens)
		r.PUT("/tokens", h.Admin.SiteRecon.SetToken)
		r.POST("/tokens/delete", h.Admin.SiteRecon.DeleteToken)
	}
}
