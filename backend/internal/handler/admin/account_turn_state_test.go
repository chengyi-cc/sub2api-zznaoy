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

func TestTriggerTurnStateAcquisitionRejectsInvalidInput(test *testing.T) {
	gin.SetMode(gin.TestMode)
	handler := &AccountHandler{}
	router := gin.New()
	router.POST("/accounts/:id/turn-state/acquire", handler.TriggerTurnStateAcquisition)
	for _, input := range []struct{ id, body string }{
		{"invalid", `{"model":"gpt-6-astra"}`}, {"0", `{"model":"gpt-6-astra"}`},
		{"42", `{}`}, {"42", `{"model":123}`}, {"42", `{"model":" "}`},
		{"42", `{"model":"` + strings.Repeat("a", 4200) + `"}`},
	} {
		request := httptest.NewRequest(http.MethodPost, "/accounts/"+input.id+"/turn-state/acquire", strings.NewReader(input.body))
		request.Header.Set("Content-Type", "application/json")
		result := httptest.NewRecorder()
		router.ServeHTTP(result, request)
		require.Equal(test, http.StatusBadRequest, result.Code)
	}
}
