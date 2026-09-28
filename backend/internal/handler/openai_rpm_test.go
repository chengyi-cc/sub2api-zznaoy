package handler

import (
	"errors"
	"fmt"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestOpenAIRPMAdmissionPreservesExhaustionAcrossSelection(t *testing.T) {
	a := openAIRPMAdmission{exhausted: true}
	err := a.selectionError(fmt.Errorf("empty pool: %w", service.ErrNoAvailableAccounts))
	require.ErrorIs(t, err, service.ErrOpenAIRPMExhausted)
	cls := classifySelectionFailureError(err, noAccountErrorClassification{})
	require.Equal(t, 429, cls.Status)
	require.Contains(t, cls.Message, "per-minute")
	dbErr := errors.New("database unavailable")
	require.ErrorIs(t, a.selectionError(dbErr), dbErr)
	require.ErrorIs(t, a.selectionError(service.ErrOpenAIRPMUnavailable), service.ErrOpenAIRPMUnavailable)
}

func TestOpenAIRPMRetryAfterOnlyOnExhaustedResponse(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	a := openAIRPMAdmission{exhausted: true, resetAt: time.Now().Add(20 * time.Second)}
	a.retryAfter(c, nil)
	require.Empty(t, w.Header().Get("Retry-After"), "successful failover must not retain a rate-limit header")
	a.retryAfter(c, service.ErrOpenAIRPMUnavailable)
	require.Empty(t, w.Header().Get("Retry-After"))
	a.retryAfter(c, service.ErrOpenAIRPMExhausted)
	seconds, err := strconv.Atoi(w.Header().Get("Retry-After"))
	require.NoError(t, err)
	require.GreaterOrEqual(t, seconds, 1)
	require.LessOrEqual(t, seconds, 20)
}

func TestOpenAIRPMRetryRestoresOnlyLocallyExhaustedOverflowAccounts(t *testing.T) {
	excluded := map[int64]struct{}{1: {}, 2: {}}
	a := openAIRPMAdmission{excluded: excluded, overflowExcluded: map[int64]struct{}{1: {}}}
	require.False(t, a.retryOverflowSelection(service.ErrOpenAIRPMUnavailable))
	require.True(t, a.retryOverflowSelection(service.ErrNoAvailableAccounts))
	require.NotContains(t, excluded, int64(1))
	require.Contains(t, excluded, int64(2))
	require.False(t, a.retryOverflowSelection(service.ErrNoAvailableAccounts))
}
