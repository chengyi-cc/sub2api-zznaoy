package admin

import (
	"github.com/Wei-Shaw/sub2api/internal/pkg/response"
	"github.com/gin-gonic/gin"
	"net/http"
)

// Both routes are registered only within the authenticated admin route group.
func (h *OpenAIOAuthReauthHandler) CredentialEncryption(c *gin.Context) {
	c.Header("Cache-Control", "no-store")
	status, err := h.service.CredentialEncryptionStatus()
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, status)
}

func (h *OpenAIOAuthReauthHandler) ResetCredentialEncryption(c *gin.Context) {
	c.Header("Cache-Control", "no-store")
	var input struct {
		DiscardSavedLoginCredentials bool `json:"discard_saved_login_credentials"`
	}
	if err := c.ShouldBindJSON(&input); err != nil || !input.DiscardSavedLoginCredentials {
		response.ErrorWithDetails(c, http.StatusBadRequest, "Explicitly confirm discarding saved re-login credentials", "CREDENTIAL_RECOVERY_CONFIRM_REQUIRED", nil)
		return
	}
	status, err := h.service.ResetCredentialEncryption(c.Request.Context())
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, status)
}

func (h *OpenAIOAuthReauthHandler) InitializeCredentialEncryption(c *gin.Context) {
	c.Header("Cache-Control", "no-store")
	status, err := h.service.InitializeCredentialEncryption()
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, status)
}
