package admin

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestTurnStateSettingsRejectsInvalidPayloadWithoutEchoingSecrets(test *testing.T) {
	gin.SetMode(gin.TestMode)
	handler := &AccountHandler{}
	router := gin.New()
	router.PUT("/admin/accounts/turn-state/settings", handler.SaveTurnStateSettings)
	for _, payload := range []string{`{"proxy_password":"private-secret",`, `{"proxy_password":"` + strings.Repeat("private-secret", 12000) + `"}`} {
		request := httptest.NewRequest(http.MethodPut, "/admin/accounts/turn-state/settings", strings.NewReader(payload))
		request.Header.Set("Content-Type", "application/json")
		result := httptest.NewRecorder()
		router.ServeHTTP(result, request)
		require.Equal(test, http.StatusBadRequest, result.Code)
		require.NotContains(test, result.Body.String(), "private-secret")
	}
}
