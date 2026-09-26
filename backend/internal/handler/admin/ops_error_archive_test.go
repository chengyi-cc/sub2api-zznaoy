package admin

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/errorarchive"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestOpsArchiveDownloadReturnsAttachmentAndHonorsMonitoringSwitch(t *testing.T) {
	gin.SetMode(gin.TestMode)
	cfg := &config.Config{}
	cfg.JWT.Secret = strings.Repeat("test-stable-key", 3)
	cfg.Ops.Enabled = true
	cfg.Ops.ErrorArchive = config.OpsErrorArchiveConfig{Enabled: true, Directory: t.TempDir(), RetentionHours: 72, MaxRequestKB: 4, MaxDiskMB: 16}
	ops := service.NewOpsService(nil, nil, cfg, nil, nil, nil, nil, nil, nil, nil, nil)
	capture, enabled := ops.CaptureErrorRequest(context.Background(), io.NopCloser(strings.NewReader("synthetic input")))
	require.True(t, enabled)
	_, err := io.ReadAll(capture)
	require.NoError(t, err)
	ref := capture.Save(&errorarchive.Entry{ContentLength: 15, APIKeyID: 1})
	ops.StopRuntimeSettingsRefresh()
	handler := NewOpsHandler(ops)
	router := gin.New()
	// Production mounts this handler exclusively below its shared admin-auth group.
	router.GET("/archives/:archive_id", handler.GetErrorArchive)
	request := func(id string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, httptest.NewRequest("GET", "/archives/"+id, nil))
		return rec
	}
	rec := request(ref.ID)
	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, "no-store", rec.Header().Get("Cache-Control"))
	require.Contains(t, rec.Header().Get("Content-Disposition"), "attachment; filename=error-capture-")
	require.Contains(t, rec.Body.String(), "request_wire_base64")
	require.Equal(t, http.StatusNotFound, request("invalid-id").Code)
	cfg.Ops.Enabled = false
	require.Equal(t, http.StatusNotFound, request(ref.ID).Code)
}
