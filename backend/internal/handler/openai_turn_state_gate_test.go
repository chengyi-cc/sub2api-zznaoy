package handler

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestTurnStateGateExhaustionReturns503WithoutRateLimitMessage(test *testing.T) {
	for _, anthropic := range []bool{false, true} {
		recorder := httptest.NewRecorder()
		requestContext, _ := gin.CreateTestContext(recorder)
		requestContext.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
		failure := &service.UpstreamFailoverError{StatusCode: 503, Reason: service.OpenAITurnStateUnavailableReason, ResponseHeaders: http.Header{"Retry-After": []string{"5"}}}
		handler := &OpenAIGatewayHandler{}
		if anthropic {
			handler.handleAnthropicFailoverExhausted(requestContext, failure, false)
		} else {
			handler.handleFailoverExhausted(requestContext, failure, false)
		}
		require.Equal(test, 503, recorder.Code)
		require.Equal(test, "5", recorder.Header().Get("Retry-After"))
		require.Contains(test, recorder.Body.String(), service.OpenAITurnStateUnavailableMessage)
		require.NotContains(test, recorder.Body.String(), "rate_limit")
	}
}
