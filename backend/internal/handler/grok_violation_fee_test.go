//go:build unit

package handler

import (
	"net/http/httptest"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestClearGrokViolationFeeTurnStateResetsMarkAndGuard(t *testing.T) {
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	service.MarkGrokViolationFee(c)
	c.Set(grokViolationFeeRecordedKey, true)

	clearGrokViolationFeeTurnState(c)

	require.False(t, service.HasGrokViolationFee(c))
	require.False(t, c.GetBool(grokViolationFeeRecordedKey))
	clearGrokViolationFeeTurnState(nil)
}
