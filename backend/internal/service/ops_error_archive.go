package service

import (
	"context"
	"crypto/sha256"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/errorarchive"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
)

func (s *OpsService) initErrorArchive() {
	if s.cfg == nil || !s.cfg.Ops.Enabled || !s.cfg.Ops.ErrorArchive.Enabled {
		return
	}
	c := s.cfg.Ops.ErrorArchive
	if s.cfg.JWT.Secret == "" {
		logger.LegacyPrintf("ops.error_archive", "disabled: stable server encryption secret unavailable")
		return
	}
	dir := c.Directory
	if dir == "" {
		dir = filepath.Join(os.Getenv("DATA_DIR"), "data", "error-archives")
		if os.Getenv("DATA_DIR") != "" {
			dir = filepath.Join(os.Getenv("DATA_DIR"), "error-archives")
		}
	}
	key := sha256.Sum256([]byte("sub2api/error-archive/v1\x00" + s.cfg.JWT.Secret))
	store, err := errorarchive.New(errorarchive.Config{Directory: dir, Key: key[:], Retention: time.Duration(c.RetentionHours) * time.Hour, MaxBytes: int64(c.MaxDiskMB) << 20, CaptureBytes: c.MaxRequestKB << 10})
	if err != nil {
		logger.LegacyPrintf("ops.error_archive", "initialization failed: error_type=%T", err)
		return
	}
	s.errorArchive = store
}

func (s *OpsService) CaptureErrorRequest(ctx context.Context, body io.ReadCloser) (*errorarchive.Capture, bool) {
	if s == nil || s.errorArchive == nil || !s.IsMonitoringEnabled(ctx) {
		return nil, false
	}
	return s.errorArchive.Capture(body), true
}
func (s *OpsService) ReadErrorArchive(ctx context.Context, id string) (*errorarchive.Entry, error) {
	if s == nil || s.errorArchive == nil {
		return nil, os.ErrNotExist
	}
	if err := s.RequireMonitoringEnabled(ctx); err != nil {
		return nil, err
	}
	return s.errorArchive.Read(id)
}
func (s *OpsService) ErrorArchiveStats() map[string]any {
	if s == nil || s.errorArchive == nil {
		return map[string]any{"enabled": false}
	}
	stats := s.errorArchive.Stats()
	stats["enabled"] = true
	return stats
}
