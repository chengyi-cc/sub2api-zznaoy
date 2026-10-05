package admin

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestCredentialRecoveryRequiresExplicitDiscard(t *testing.T) {
	for _, body := range []string{"", "{}", `{"discard_saved_login_credentials":false}`, `{"discard_saved_login_credentials":"true"}`} {
		t.Run(body, func(t *testing.T) {
			router := gin.New()
			h := &OpenAIOAuthReauthHandler{}
			router.POST("/reset", h.ResetCredentialEncryption)
			req := httptest.NewRequest(http.MethodPost, "/reset", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)
			require.Equal(t, http.StatusBadRequest, w.Code)
			require.Contains(t, w.Body.String(), "CREDENTIAL_RECOVERY_CONFIRM_REQUIRED")
			require.Equal(t, "no-store", w.Header().Get("Cache-Control"))
		})
	}
}
