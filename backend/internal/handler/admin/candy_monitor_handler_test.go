//go:build unit

package admin

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type candyFilterCapture struct {
	service.CandyMonitorRepository
	filter service.CandyMonitorFilter
	called bool
}

func (r *candyFilterCapture) List(_ context.Context, f service.CandyMonitorFilter) ([]service.CandyMonitorAccount, int64, error) {
	r.filter, r.called = f, true
	return []service.CandyMonitorAccount{}, 0, nil
}

func TestCandyMonitorListFilters(t *testing.T) {
	for _, tc := range []struct {
		query  string
		status int
		check  func(*testing.T, service.CandyMonitorFilter)
	}{
		{"?enabled=false&group_id=8&platform=openai&type=oauth&status=rate_limited&privacy_mode=training_off&verdict=incorrect&search=team&page=2&page_size=100", http.StatusOK, func(t *testing.T, f service.CandyMonitorFilter) {
			require.NotNil(t, f.Enabled)
			require.False(t, *f.Enabled)
			require.EqualValues(t, 8, f.GroupID)
			require.Equal(t, "openai", f.Platform)
			require.Equal(t, "oauth", f.Type)
			require.Equal(t, "rate_limited", f.Status)
			require.Equal(t, "training_off", f.PrivacyMode)
			require.Equal(t, "incorrect", f.Verdict)
			require.Equal(t, "team", f.Search)
			require.Equal(t, 2, f.Page)
			require.Equal(t, 100, f.PageSize)
		}},
		{"?enabled=true&ungrouped=true", http.StatusOK, func(t *testing.T, f service.CandyMonitorFilter) {
			require.NotNil(t, f.Enabled)
			require.True(t, *f.Enabled)
			require.True(t, f.Ungrouped)
		}},
		{"?enabled_only=true", http.StatusOK, func(t *testing.T, f service.CandyMonitorFilter) {
			require.Nil(t, f.Enabled)
			require.True(t, f.EnabledOnly)
		}},
		{"", http.StatusOK, func(t *testing.T, f service.CandyMonitorFilter) {
			require.Nil(t, f.Enabled)
			require.False(t, f.EnabledOnly)
		}},
		{"?enabled=invalid", http.StatusBadRequest, nil},
		{"?group_id=-1", http.StatusBadRequest, nil},
	} {
		t.Run(tc.query, func(t *testing.T) {
			repo := &candyFilterCapture{}
			svc := service.NewCandyMonitorService(repo, nil, &service.AccountTestService{})
			defer svc.Stop()
			handler := NewCandyMonitorHandler(svc)
			router := gin.New()
			router.GET("/accounts", handler.List)
			w := httptest.NewRecorder()
			router.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/accounts"+tc.query, nil))
			require.Equal(t, tc.status, w.Code)
			if tc.check != nil {
				require.True(t, repo.called)
				tc.check(t, repo.filter)
			} else {
				require.False(t, repo.called)
			}
		})
	}
}
