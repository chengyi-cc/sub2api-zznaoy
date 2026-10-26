package admin

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type giftGenerateAdminStub struct {
	service.AdminService
	calls int
	input *service.GenerateRedeemCodesInput
}

func (s *giftGenerateAdminStub) GenerateRedeemCodes(_ context.Context, in *service.GenerateRedeemCodesInput) ([]service.RedeemCode, error) {
	s.calls++
	s.input = in
	return []service.RedeemCode{{ID: 1, Code: "paid", Value: in.Value}, {ID: 2, Code: "gift", Value: in.Value / 100, IsGift: true}}, nil
}
func (s *giftGenerateAdminStub) GetRedeemCode(_ context.Context, id int64) (*service.RedeemCode, error) {
	if id == 1 {
		return &service.RedeemCode{ID: 1, Code: "paid", Value: s.input.Value}, nil
	}
	return &service.RedeemCode{ID: 2, Code: "gift", Value: s.input.Value / 100, IsGift: true}, nil
}

func TestRedeemGiftGenerate_ReplaysAndRejectsConflictingRetry(t *testing.T) {
	gin.SetMode(gin.TestMode)
	cfg := service.DefaultIdempotencyConfig()
	cfg.ObserveOnly = false
	cfg.MaxStoredResponseLen = 32 // references fit; full code DTOs would be truncated
	service.SetDefaultIdempotencyCoordinator(service.NewIdempotencyCoordinator(newMemoryIdempotencyRepoStub(), cfg))
	t.Cleanup(func() { service.SetDefaultIdempotencyCoordinator(nil) })
	svc := &giftGenerateAdminStub{}
	router := gin.New()
	router.POST("/generate", NewRedeemHandler(svc, nil).Generate)
	post := func(key, body string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/generate", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Idempotency-Key", key)
		router.ServeHTTP(rec, req)
		return rec
	}
	// Client-supplied gift values and identity are not accepted as generation inputs.
	body := `{"count":1,"type":"balance","value":10,"include_gift":true,"gift_value":99999,"is_gift":false}`
	first := post("gift-operation-one", body)
	require.Equal(t, http.StatusOK, first.Code, first.Body.String())
	second := post("gift-operation-one", body)
	require.Equal(t, http.StatusOK, second.Code)
	require.JSONEq(t, first.Body.String(), second.Body.String())
	require.Equal(t, "true", second.Header().Get("X-Idempotency-Replayed"))
	require.Equal(t, 1, svc.calls)
	require.True(t, svc.input.IncludeGift)
	require.Equal(t, 10.0, svc.input.Value)
	conflict := post("gift-operation-one", strings.Replace(body, `"value":10`, `"value":20`, 1))
	require.Equal(t, http.StatusConflict, conflict.Code)
	require.Equal(t, 1, svc.calls)
	for _, key := range []string{"", "invalid key"} {
		rec := post(key, body)
		require.Equal(t, http.StatusBadRequest, rec.Code)
	}
	require.Equal(t, 1, svc.calls)
	service.SetDefaultIdempotencyCoordinator(nil)
	unavailable := post("gift-operation-two", body)
	require.Equal(t, http.StatusServiceUnavailable, unavailable.Code)
	require.Equal(t, 1, svc.calls)
}
