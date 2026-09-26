package handler

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/errorarchive"
	"github.com/Wei-Shaw/sub2api/internal/pkg/httputil"
	middleware2 "github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func archiveTestService(t *testing.T) *service.OpsService {
	t.Helper()
	cfg := &config.Config{}
	cfg.JWT.Secret = strings.Repeat("private-server-key", 3)
	cfg.Ops.Enabled = true
	cfg.Ops.ErrorArchive = config.OpsErrorArchiveConfig{Enabled: true, Directory: t.TempDir(), RetentionHours: 72, MaxRequestKB: 4, MaxDiskMB: 16}
	ops := service.NewOpsService(nil, nil, cfg, nil, nil, nil, nil, nil, nil, nil, nil)
	t.Cleanup(ops.StopRuntimeSettingsRefresh)
	return ops
}

func TestOpsErrorArchiveCapturesWireBodyAndReadCause(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, badGzip := range []bool{false, true} {
		ops := archiveTestService(t)
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader("private body"))
		c.Request.Header.Set("Authorization", "Bearer HEADER_MUST_NOT_BE_CAPTURED")
		if badGzip {
			c.Request.Header.Set("Content-Encoding", "gzip")
		}
		c.Set(string(middleware2.ContextKeyAPIKey), &service.APIKey{ID: 42})
		finish := beginOpsErrorArchive(c, ops)
		require.NotNil(t, finish)
		_, err := httputil.ReadRequestBodyWithPrealloc(c.Request)
		if badGzip {
			require.Error(t, err)
		} else {
			require.NoError(t, err)
		}
		ref := finish(400, []byte("request failed"), false)
		ops.StopRuntimeSettingsRefresh()
		entry, err := ops.ReadErrorArchive(context.Background(), ref.ID)
		require.NoError(t, err)
		require.Equal(t, "private body", string(entry.Request))
		require.Equal(t, int64(42), entry.APIKeyID)
		if badGzip {
			require.Equal(t, "decode_content_encoding", entry.ReadError)
			require.False(t, entry.Complete)
			require.Equal(t, "gzip", entry.ContentEncoding)
		} else {
			require.True(t, entry.Complete)
		}
		raw, err := json.Marshal(entry)
		require.NoError(t, err)
		require.NotContains(t, string(raw), "HEADER_MUST_NOT_BE_CAPTURED")
	}
}

func TestOpsErrorArchiveDiscardsHealthyAndUnauthenticatedRequests(t *testing.T) {
	for _, tc := range []struct {
		authenticated bool
		status        int
	}{{true, 200}, {false, 400}, {true, 401}, {true, 403}} {
		ops := archiveTestService(t)
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader("body"))
		if tc.authenticated {
			c.Set(string(middleware2.ContextKeyAPIKey), &service.APIKey{ID: 42})
		}
		finish := beginOpsErrorArchive(c, ops)
		_, err := io.ReadAll(c.Request.Body)
		require.NoError(t, err)
		require.Empty(t, finish(tc.status, nil, false).State)
		ops.StopRuntimeSettingsRefresh()
		require.Equal(t, uint64(0), ops.ErrorArchiveStats()["saved"])
		require.Equal(t, 0, ops.ErrorArchiveStats()["in_flight"])
	}
}

func TestOpsErrorArchiveMiddlewareReleasesSlotOnPanic(t *testing.T) {
	ops := archiveTestService(t)
	router := gin.New()
	router.Use(gin.CustomRecovery(func(c *gin.Context, recovered any) { c.AbortWithStatus(500) }))
	router.Use(OpsErrorLoggerMiddleware(ops))
	router.POST("/v1/responses", func(c *gin.Context) { panic("synthetic panic") })
	for i := 0; i < 10; i++ {
		router.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/v1/responses", strings.NewReader("body")))
	}
	require.Equal(t, 0, ops.ErrorArchiveStats()["in_flight"])
	require.Equal(t, uint64(0), ops.ErrorArchiveStats()["skipped_capacity"])
}

func TestOpsErrorArchiveReferenceSurvivesPersistenceLimits(t *testing.T) {
	ref := errorarchive.Ref{ID: strings.Repeat("a", 32), State: "queued"}
	for _, body := range []string{strings.Repeat("\\\"\n", 20000), "{\"error\":\"" + strings.Repeat("x", 20000) + "\"}"} {
		stored, _ := service.SanitizeOpsErrorBodyForQueue(withOpsArchiveRef(body, ref))
		var decoded map[string]any
		require.NoError(t, json.Unmarshal([]byte(stored), &decoded))
		require.Equal(t, ref.ID, decoded["diagnostic_archive"].(map[string]any)["id"])
	}
}

func TestOpsErrorArchiveRecoveredCorrectionHasDownloadReference(t *testing.T) {
	setupOpsErrorLogTestQueue(t, 4)
	ops := archiveTestService(t)
	router := gin.New()
	router.Use(OpsErrorLoggerMiddleware(ops))
	router.POST("/v1/responses", func(c *gin.Context) {
		c.Set(string(middleware2.ContextKeyAPIKey), &service.APIKey{ID: 42})
		_, _ = io.ReadAll(c.Request.Body)
		errorarchive.AddDiagnostic(c.Request.Context(), "tool_validation", []byte("original"), []byte("invalid wrapper"))
		c.JSON(200, gin.H{"status": "completed"})
	})
	router.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/v1/responses", strings.NewReader("body")))
	require.Len(t, opsErrorLogQueue, 1)
	job := <-opsErrorLogQueue
	require.Equal(t, 200, job.entry.StatusCode)
	var decoded map[string]any
	require.NoError(t, json.Unmarshal([]byte(job.entry.ErrorBody), &decoded))
	ref := decoded["diagnostic_archive"].(map[string]any)
	ops.StopRuntimeSettingsRefresh()
	entry, err := ops.ReadErrorArchive(context.Background(), ref["id"].(string))
	require.NoError(t, err)
	require.NotEmpty(t, entry.Diagnostics)
}
