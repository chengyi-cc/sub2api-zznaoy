//go:build unit

package routes

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/handler"
	adminhandler "github.com/Wei-Shaw/sub2api/internal/handler/admin"
	"github.com/Wei-Shaw/sub2api/internal/pkg/adminaccess"
	"github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type redeemRouteUserRepo struct {
	service.UserRepository
	users map[int64]*service.User
}

func (r *redeemRouteUserRepo) GetByID(_ context.Context, id int64) (*service.User, error) {
	if u := r.users[id]; u != nil {
		clone := *u
		return &clone, nil
	}
	return nil, service.ErrUserNotFound
}

func (r *redeemRouteUserRepo) GetUserAvatar(_ context.Context, _ int64) (*service.UserAvatar, error) {
	return nil, nil
}

// Exercise the production route registration and real authentication middleware,
// not a mock middleware that unconditionally rejects requests.
func TestRedeemRoutesRequireRealAdminAuthorization(t *testing.T) {
	gin.SetMode(gin.TestMode)
	cfg := &config.Config{JWT: config.JWTConfig{Secret: "redeem-route-test-secret-only", ExpireHour: 1}}
	auth := service.NewAuthService(nil, nil, nil, nil, cfg, nil, nil, nil, nil, nil, nil, nil, nil)
	repo := &redeemRouteUserRepo{users: map[int64]*service.User{
		1: {ID: 1, Role: service.RoleUser, Status: service.StatusActive, TokenVersion: 1},
		2: {ID: 2, Role: service.RoleAdmin, Status: service.StatusActive, TokenVersion: 2},
		3: {ID: 3, Role: service.RoleAdmin, Status: service.StatusDisabled, TokenVersion: 1},
	}}
	userSvc := service.NewUserService(repo, nil, nil, nil)
	router := gin.New()
	handlers := &handler.Handlers{Admin: &handler.AdminHandlers{
		Account: &adminhandler.AccountHandler{}, OpenAIOAuth: &adminhandler.OpenAIOAuthHandler{},
		Redeem: &adminhandler.RedeemHandler{},
	}}
	RegisterAdminRoutes(router.Group("/api/v1"), handlers,
		middleware.NewAdminAuthMiddleware(auth, userSvc, nil, nil),
		middleware.AuditLogMiddleware(func(c *gin.Context) { c.Next() }),
		middleware.StepUpAuthMiddleware(func(c *gin.Context) { c.Next() }), nil, nil)
	token := func(id int64, role string, version int64) string {
		value, err := auth.GenerateToken(context.Background(), &service.User{ID: id, Role: role, TokenVersion: version})
		require.NoError(t, err)
		return "Bearer " + value
	}
	for _, actor := range []struct {
		name, authorization string
		status              int
	}{
		{"anonymous", "", http.StatusUnauthorized},
		{"invalid token", "Bearer not-a-token", http.StatusUnauthorized},
		{"ordinary user", token(1, service.RoleUser, 1), http.StatusForbidden},
		{"admin claim but ordinary database role", token(1, service.RoleAdmin, 1), http.StatusForbidden},
		{"disabled admin", token(3, service.RoleAdmin, 1), http.StatusUnauthorized},
		{"revoked admin token", token(2, service.RoleAdmin, 1), http.StatusUnauthorized},
	} {
		t.Run(actor.name, func(t *testing.T) {
			for _, endpoint := range []struct{ method, path string }{
				{http.MethodPost, "/generate"}, {http.MethodPost, "/batch-update"},
				{http.MethodPost, "/create-and-redeem"}, {http.MethodGet, "/export"},
				{http.MethodGet, "/1"}, {http.MethodPost, "/1/expire"},
			} {
				request := httptest.NewRequest(endpoint.method, "/api/v1/admin/redeem-codes"+endpoint.path,
					strings.NewReader(`{"count":1,"type":"balance","value":10,"include_gift":true}`))
				request.Header.Set("Content-Type", "application/json")
				request.Header.Set("Authorization", actor.authorization)
				response := httptest.NewRecorder()
				router.ServeHTTP(response, request)
				require.Equal(t, actor.status, response.Code, endpoint.path+": "+response.Body.String())
			}
		})
	}
	// A genuine current admin gets to handler validation (400), proving the
	// routes are present and the test isn't passing merely because of a 404.
	request := httptest.NewRequest(http.MethodPost, "/api/v1/admin/redeem-codes/generate", strings.NewReader(`{"count":0,"type":"balance"}`))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", token(2, service.RoleAdmin, 2))
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	require.Equal(t, http.StatusBadRequest, response.Code, response.Body.String())
}

type mergedScopedKeyRepo struct {
	service.ScopedAdminKeyRepository
	key *service.ScopedAdminKey
}

func (r *mergedScopedKeyRepo) InsertScopedAdminKey(_ context.Context, key *service.ScopedAdminKey) error {
	r.key = key
	return nil
}
func (r *mergedScopedKeyRepo) GetScopedAdminKey(context.Context, string) (*service.ScopedAdminKey, error) {
	return r.key, nil
}
func (r *mergedScopedKeyRepo) TouchScopedAdminKey(context.Context, string, string, time.Time, string, time.Time) error {
	return nil
}

// Upstream scoped keys must not implicitly gain access to this fork's financial
// or credential-management endpoints, even when every catalog permission is on.
func TestMergedScopedAdminKeysCannotAccessCustomSensitiveRoutes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	policy := adminaccess.Policy{}
	for _, permission := range adminaccess.PermissionCatalog() {
		if permission.Scope == "global" {
			policy.GlobalPermissions = append(policy.GlobalPermissions, permission.ID)
		} else {
			policy.Rules = append(policy.Rules, adminaccess.Rule{Permissions: []string{permission.ID}, AllGroups: true, IncludeUngrouped: true, AccountTypes: []string{"oauth", "apikey"}})
		}
	}
	settings := service.NewSettingService(nil, nil)
	settings.SetScopedAdminKeyRepository(&mergedScopedKeyRepo{})
	key, _, err := settings.CreateScopedAdminKey(context.Background(), service.ScopedAdminKeyInput{Name: "merge-regression", Enabled: true, Policy: policy}, 9)
	require.NoError(t, err)
	users := service.NewUserService(&redeemRouteUserRepo{users: map[int64]*service.User{9: {ID: 9, Role: service.RoleAdmin, Status: service.StatusActive}}}, nil, nil, nil)
	router := gin.New()
	handlers := &handler.Handlers{Admin: &handler.AdminHandlers{
		Account: &adminhandler.AccountHandler{}, OpenAIOAuth: &adminhandler.OpenAIOAuthHandler{}, Redeem: &adminhandler.RedeemHandler{},
	}}
	RegisterAdminRoutes(router.Group("/api/v1"), handlers, middleware.NewAdminAuthMiddleware(nil, users, settings, nil),
		middleware.AuditLogMiddleware(func(c *gin.Context) { c.Next() }),
		middleware.StepUpAuthMiddleware(func(c *gin.Context) { c.Next() }), settings, nil)
	for _, endpoint := range []struct{ method, path string }{
		{"POST", "/redeem-codes/generate"}, {"POST", "/redeem-codes/create-and-redeem"},
		{"POST", "/redeem-codes/batch-update"}, {"GET", "/redeem-codes/export"},
		{"GET", "/invoice/requests"}, {"POST", "/invoice/requests/1/issue"},
		{"PUT", "/settings/openai-account-template"},
		{"PUT", "/astra-borrow"}, {"POST", "/prism-runtime/restart"},
	} {
		req := httptest.NewRequest(endpoint.method, "/api/v1/admin"+endpoint.path, nil)
		req.Header.Set("x-api-key", key)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusForbidden, rec.Code, endpoint.path+": "+rec.Body.String())
	}
}
