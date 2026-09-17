package admin

import (
	"net/http"
	"strconv"

	"github.com/Wei-Shaw/sub2api/internal/pkg/response"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
)

func (handler *AccountHandler) GetTurnStateSettings(requestContext *gin.Context) {
	settings, err := handler.turnStateGateway.GetTurnStateSettings(requestContext.Request.Context())
	if err != nil {
		response.ErrorFrom(requestContext, err)
		return
	}
	response.Success(requestContext, settings)
}

func (handler *AccountHandler) SaveTurnStateSettings(requestContext *gin.Context) {
	requestContext.Request.Body = http.MaxBytesReader(requestContext.Writer, requestContext.Request.Body, 128<<10)
	var settings service.TurnStateSettingsView
	if requestContext.ShouldBindJSON(&settings) != nil {
		response.BadRequest(requestContext, "采集配置格式无效或超过128KB")
		return
	}
	updated, err := handler.turnStateGateway.SaveTurnStateSettings(requestContext.Request.Context(), settings)
	if err != nil {
		response.ErrorFrom(requestContext, err)
		return
	}
	response.Success(requestContext, updated)
}

func (handler *AccountHandler) TriggerTurnStateAcquisition(requestContext *gin.Context) {
	accountID, err := strconv.ParseInt(requestContext.Param("id"), 10, 64)
	if err != nil || accountID <= 0 {
		response.BadRequest(requestContext, "Invalid account ID")
		return
	}
	var request struct {
		Model string `json:"model" binding:"required"`
	}
	requestContext.Request.Body = http.MaxBytesReader(requestContext.Writer, requestContext.Request.Body, 4096)
	if err := requestContext.ShouldBindJSON(&request); err != nil {
		response.BadRequest(requestContext, "请填写有效模型名称")
		return
	}
	if err := handler.turnStateGateway.TriggerTurnStateAcquisition(requestContext.Request.Context(), accountID, request.Model); err != nil {
		response.ErrorFrom(requestContext, err)
		return
	}
	response.Success(requestContext, gin.H{"queued": true, "model": request.Model})
}

func (handler *AccountHandler) GetTurnStateAutoStatus(requestContext *gin.Context) {
	accountID, err := strconv.ParseInt(requestContext.Param("id"), 10, 64)
	if err != nil || accountID <= 0 {
		response.BadRequest(requestContext, "Invalid account ID")
		return
	}
	account, err := handler.adminService.GetAccount(requestContext.Request.Context(), accountID)
	if err != nil {
		response.ErrorFrom(requestContext, err)
		return
	}
	status := handler.turnStateGateway.TurnStateAutoStatus(requestContext.Request.Context(), account, requestContext.Query("history") == "1")
	response.Success(requestContext, status)
}
