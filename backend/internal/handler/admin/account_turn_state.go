package admin

import (
	"strconv"

	"github.com/Wei-Shaw/sub2api/internal/pkg/response"
	"github.com/gin-gonic/gin"
)

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
