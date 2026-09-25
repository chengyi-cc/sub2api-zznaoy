package service

import (
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
)

// Migration-only cleanup of the retired local image implementation. No new
// images are written here and no HTTP route can serve these files anymore.
func excelBPSLegacyImageRoot(cfg *config.Config) string {
	if cfg == nil {
		return ""
	}
	root := strings.TrimSpace(cfg.Gateway.ImageJobs.RootDir)
	if root == "" {
		base := strings.TrimSpace(cfg.Pricing.DataDir)
		if base == "" {
			base = "./data"
		}
		root = filepath.Join(base, "jobs", "images")
	}
	return filepath.Clean(root) + "-excel-inputs"
}

func cleanupLegacyExcelImages(root string, now time.Time) error {
	if root == "" {
		return nil
	}
	entries, err := os.ReadDir(root)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	for _, entry := range entries {
		name := entry.Name()
		known := len(name) == 64 && strings.Trim(name, "0123456789abcdef") == ""
		if !known && !strings.HasPrefix(name, ".pending-") {
			continue
		}
		info, err := entry.Info()
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() || now.Before(info.ModTime().Add(5*time.Minute)) {
			continue
		}
		if err := os.Remove(filepath.Join(root, name)); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

func (s *ExcelBPSImageService) Start() {
	s.startOnce.Do(func() {
		go func() {
			defer close(s.done)
			ticker := time.NewTicker(time.Minute)
			defer ticker.Stop()
			for {
				select {
				case <-s.stop:
					return
				default:
				}
				if err := cleanupLegacyExcelImages(s.legacyRoot, time.Now()); err != nil {
					logger.L().Warn("excel_bps legacy image cleanup failed; will retry")
				}
				select {
				case <-s.stop:
					return
				case <-ticker.C:
				}
			}
		}()
	})
}

func (s *ExcelBPSImageService) Stop() {
	if s == nil {
		return
	}
	s.stopOnce.Do(func() { close(s.stop) })
	s.Start()
	<-s.done
}
