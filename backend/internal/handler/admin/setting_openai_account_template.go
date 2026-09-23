package admin

import (
	"errors"
	"github.com/Wei-Shaw/sub2api/internal/pkg/response"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"net/http"
)

func (h *SettingHandler) GetOpenAIAccountTemplate(c *gin.Context) {
	v, err := h.settingService.GetOpenAIAccountTemplate(c.Request.Context())
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}
	response.Success(c, v)
}
func (h *SettingHandler) SaveOpenAIAccountTemplate(c *gin.Context) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 65536)
	var v service.OpenAIAccountTemplate
	if err := c.ShouldBindJSON(&v); err != nil {
		response.BadRequest(c, "Invalid account template")
		return
	}
	if err := h.settingService.SaveOpenAIAccountTemplate(c.Request.Context(), &v); err != nil {
		if errors.Is(err, service.ErrOpenAIAccountTemplateInvalid) {
			response.BadRequest(c, err.Error())
		} else {
			response.ErrorFrom(c, err)
		}
		return
	}
	response.Success(c, v)
}
