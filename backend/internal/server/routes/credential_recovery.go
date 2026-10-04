package routes

import (
	"github.com/Wei-Shaw/sub2api/internal/handler"
	"github.com/gin-gonic/gin"
	"net/http"
)

func registerOpenAIOAuthReauthWorkerRoutes(v1 *gin.RouterGroup, h *handler.Handlers) {
	worker := v1.Group("/internal/openai-reauth")
	worker.Use(func(c *gin.Context) {
		c.Header("Cache-Control", "no-store")
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 1<<20)
	})
	worker.POST("/claim", h.Admin.OpenAIOAuthReauth.Claim)
	worker.POST("/runtime-settings", h.Admin.OpenAIOAuthReauth.RuntimeSettings)
	worker.POST("/:task_id/progress", h.Admin.OpenAIOAuthReauth.Progress)
	worker.POST("/:task_id/callback", h.Admin.OpenAIOAuthReauth.Callback)
	worker.POST("/:task_id/credentials", h.Admin.OpenAIOAuthReauth.Credentials)
	worker.POST("/:task_id/fail", h.Admin.OpenAIOAuthReauth.Fail)
}

func registerCredentialRecoveryRoutes(admin *gin.RouterGroup, h *handler.Handlers) {
	if h.Admin.AccountTokenGuardV2 == nil || h.Admin.OpenAIOAuthReauth == nil {
		return
	}
	group := admin.Group("/account-ops/token-guard-v2")
	group.Use(func(c *gin.Context) {
		c.Header("Cache-Control", "no-store")
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 16384)
	})
	group.GET("/encryption", h.Admin.OpenAIOAuthReauth.CredentialEncryption)
	group.POST("/encryption/initialize", h.Admin.OpenAIOAuthReauth.InitializeCredentialEncryption)
	group.GET("/accounts", h.Admin.AccountTokenGuardV2.List)
	group.PUT("/rules", h.Admin.AccountTokenGuardV2.SaveRules)
	group.PUT("/runtime", h.Admin.AccountTokenGuardV2.SaveRuntime)
	group.PATCH("/accounts/:id/switches", h.Admin.AccountTokenGuardV2.UpdateSwitches)
	group.POST("/accounts", h.Admin.AccountTokenGuardV2.Create)
	group.PUT("/accounts/:id", h.Admin.AccountTokenGuardV2.Update)
	group.DELETE("/accounts/:id", h.Admin.AccountTokenGuardV2.Delete)
	group.POST("/accounts/:id/probe", h.Admin.AccountTokenGuardV2.Probe)
	group.POST("/accounts/:id/relogin", h.Admin.AccountTokenGuardV2.Relogin)
}
