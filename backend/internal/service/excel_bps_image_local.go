package service

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
)

// Local inputs are separate from generated images. File modification time is a
// renewable lease, never part of the URL. Only authenticated submissions renew it.
type ExcelBPSImageService struct {
	root, secret, frontendURL string
	settings                  SettingRepository
	maxBytes                  int64
	now                       func() time.Time
	mu                        sync.Mutex
	files                     map[string]int64
	total                     int64
	startOnce, stopOnce       sync.Once
	stop, done                chan struct{}
}

func NewExcelBPSImageService(cfg *config.Config, settings SettingRepository) *ExcelBPSImageService {
	root := strings.TrimSpace(cfg.Gateway.ImageJobs.RootDir)
	if root == "" {
		base := strings.TrimSpace(cfg.Pricing.DataDir)
		if base == "" {
			base = "./data"
		}
		root = filepath.Join(base, "jobs", "images")
	}
	maxBytes := int64(cfg.Gateway.ImageJobs.MaxTotalDiskMB) << 20
	if maxBytes <= 0 {
		maxBytes = 1 << 30
	}
	return &ExcelBPSImageService{root: filepath.Clean(root) + "-excel-inputs", secret: cfg.JWT.Secret,
		frontendURL: cfg.Server.FrontendURL, settings: settings, maxBytes: maxBytes,
		now: time.Now, stop: make(chan struct{}), done: make(chan struct{})}
}

func (s *ExcelBPSImageService) publicBaseURL(ctx context.Context, requestHost string) (string, error) {
	base := ""
	if s.settings != nil {
		for _, key := range []string{SettingKeyAPIBaseURL, SettingKeyFrontendURL} {
			value, err := s.settings.GetValue(ctx, key)
			if err == nil && strings.TrimSpace(value) != "" {
				base = strings.TrimSpace(value)
				break
			}
		}
	}
	if base == "" {
		base = strings.TrimSpace(s.frontendURL)
	}
	if base == "" {
		// Use only the request authority, never arbitrary Forwarded headers.
		// The caller supplies the image itself; no other caller's images or
		// credentials are sent to this host. Prefer the admin-configured origin.
		if strings.ContainsAny(requestHost, "/\\?#@") {
			return "", fmt.Errorf("invalid public image host")
		}
		base = "https://" + requestHost
	}
	u, err := url.Parse(base)
	if err != nil || u.Scheme != "https" || u.Hostname() == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return "", fmt.Errorf("Excel BPS images require a public HTTPS site address; check the API base URL in site settings")
	}
	u.Path = strings.TrimSuffix(strings.TrimRight(u.Path, "/"), "/v1") + ExcelBPSImagePath
	u.RawPath = ""
	return u.String(), nil
}

func validExcelImageToken(token string) bool {
	if len(token) != 64 {
		return false
	}
	for _, c := range token {
		if !(c >= '0' && c <= '9') && !(c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

func (s *ExcelBPSImageService) imageToken(scope string, data []byte) string {
	mac := hmac.New(sha256.New, []byte(s.secret))
	_, _ = mac.Write([]byte("sub2api/excel-input/v1\x00" + scope + "\x00"))
	digest := sha256.Sum256(data)
	_, _ = mac.Write(digest[:])
	return hex.EncodeToString(mac.Sum(nil))
}

func (s *ExcelBPSImageService) initLocked() error {
	if s.files != nil {
		return nil
	}
	if err := os.MkdirAll(s.root, 0700); err != nil {
		return err
	}
	entries, err := os.ReadDir(s.root)
	if err != nil {
		return err
	}
	files := make(map[string]int64)
	var total int64
	for _, entry := range entries {
		name := entry.Name()
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			continue
		}
		// Track recent interrupted writes too, so later cleaner passes delete
		// them once expired instead of forgetting them after startup.
		if !validExcelImageToken(name) && !strings.HasPrefix(name, ".pending-") {
			continue
		}
		files[name] = info.Size()
		total += info.Size()
	}
	s.files, s.total = files, total
	return nil
}

func (s *ExcelBPSImageService) cleanupLocked() error {
	if err := s.initLocked(); err != nil {
		return err
	}
	var first error
	for name, size := range s.files {
		path := filepath.Join(s.root, name)
		info, err := os.Lstat(path)
		if err == nil && info.Mode().IsRegular() && s.now().Before(info.ModTime().Add(ExcelBPSImageTTL)) {
			continue
		}
		if err == nil {
			err = os.Remove(path)
		}
		if err != nil && !os.IsNotExist(err) {
			if first == nil {
				first = err
			}
			continue
		}
		delete(s.files, name)
		s.total -= size
	}
	return first
}

func (s *ExcelBPSImageService) saveImages(ctx context.Context, plan *excelBPSImagePlan, scope, base string) (map[string]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	// Production configuration always has a JWT secret. Never create guessable
	// capabilities if constructed without it (e.g. an incomplete test harness).
	if len(s.secret) < 16 {
		return nil, fmt.Errorf("missing server signing secret")
	}
	if err := s.initLocked(); err != nil {
		return nil, err
	}
	required := func() (int64, int) {
		var bytes int64
		count := 0
		for _, img := range plan.images {
			if _, ok := s.files[s.imageToken(scope, img.data)]; !ok {
				bytes += int64(len(img.data))
				count++
			}
		}
		return bytes, count
	}
	size, count := required()
	if s.total+size > s.maxBytes || len(s.files)+count > 10000 {
		if err := s.cleanupLocked(); err != nil {
			return nil, err
		}
		size, count = required()
		if s.total+size > s.maxBytes || len(s.files)+count > 10000 {
			return nil, fmt.Errorf("local image quota exceeded")
		}
	}
	links := make(map[string]string, len(plan.images))
	created := make([]string, 0, len(plan.images))
	succeeded := false
	defer func() {
		if !succeeded {
			for _, name := range created {
				if err := os.Remove(filepath.Join(s.root, name)); err == nil {
					s.total -= s.files[name]
					delete(s.files, name)
				}
			}
		}
	}()
	for _, img := range plan.images {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		token := s.imageToken(scope, img.data)
		path := filepath.Join(s.root, token)
		if _, ok := s.files[token]; !ok {
			f, err := os.CreateTemp(s.root, ".pending-")
			if err != nil {
				return nil, err
			}
			temp := f.Name()
			_, err = f.Write(img.data)
			if err == nil {
				err = f.Sync()
			}
			closeErr := f.Close()
			if err == nil {
				err = closeErr
			}
			if err == nil {
				err = os.Rename(temp, path)
			}
			if err != nil {
				_ = os.Remove(temp)
				return nil, err
			}
			s.files[token] = int64(len(img.data))
			s.total += int64(len(img.data))
			created = append(created, token)
		}
		now := s.now()
		if err := os.Chtimes(path, now, now); err != nil {
			return nil, err
		}
		links[img.placeholder] = base + "?token=" + token
	}
	succeeded = true
	return links, nil
}

// OpenImage checks the lease on every GET/HEAD; reads never extend it.
func (s *ExcelBPSImageService) OpenImage(token string) (*os.File, error) {
	if s == nil || !validExcelImageToken(token) {
		return nil, os.ErrNotExist
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	path := filepath.Join(s.root, token)
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Size() > excelBPSMaxImageBytes || !s.now().Before(info.ModTime().Add(ExcelBPSImageTTL)) {
		return nil, os.ErrNotExist
	}
	return os.Open(path)
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
				s.mu.Lock()
				err := s.cleanupLocked()
				s.mu.Unlock()
				if err != nil {
					logger.L().Warn("excel_bps local image cleanup failed; will retry")
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

// ServeExcelBPSImage is a capability-only endpoint; upstream cannot send our
// API key. Keep the token in the query so ordinary path access logs omit it.
func (s *OpenAIGatewayService) ServeExcelBPSImage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "private, no-store, max-age=0")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Referrer-Policy", "no-referrer")
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	file, err := s.excelBPSImages.OpenImage(r.URL.Query().Get("token"))
	if err != nil {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	defer file.Close()
	var header [512]byte
	n, _ := file.Read(header[:])
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", http.DetectContentType(header[:n]))
	http.ServeContent(w, r, "image", time.Time{}, file)
}
