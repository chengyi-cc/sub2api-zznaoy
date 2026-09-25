package service

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"image"
	_ "image/gif"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/google/uuid"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	_ "golang.org/x/image/webp"
)

const ExcelBPSImageTTL = 5 * time.Minute
const ExcelBPSImageObjectPrefix = "_sub2api/excel-bps-input/v1/"
const excelBPSMaxImageBytes = 20 << 20
const excelBPSMaxTotalImageBytes = 32 << 20
const excelBPSMaxImages = 16

// TemporaryImageStorage always signs links, independently of generated-image
// public URLs and TTL. Expired objects can be discovered again after a restart.
type TemporaryImageStorage interface {
	SaveTemporary(context.Context, string, string, []byte, time.Duration) (string, error)
	DeleteTemporary(context.Context, string) error
	DeleteExpiredTemporary(context.Context, time.Time) error
}

type ExcelBPSImageService struct {
	settings     *ImageStorageSettingService
	resolve      func(context.Context, bool) (TemporaryImageStorage, error)
	startOnce    sync.Once
	stopOnce     sync.Once
	stop         chan struct{}
	done         chan struct{}
	workerCtx    context.Context
	workerCancel context.CancelFunc
}

func NewExcelBPSImageService(settings *ImageStorageSettingService) *ExcelBPSImageService {
	s := &ExcelBPSImageService{settings: settings, stop: make(chan struct{}), done: make(chan struct{})}
	s.workerCtx, s.workerCancel = context.WithCancel(context.Background())
	s.resolve = s.resolveStorage
	return s
}

func (s *ExcelBPSImageService) resolveStorage(ctx context.Context, requireEnabled bool) (TemporaryImageStorage, error) {
	if s == nil || s.settings == nil || s.settings.factory == nil {
		return nil, fmt.Errorf("image storage is unavailable")
	}
	cfg, err := s.settings.effectiveConfig(ctx)
	if err != nil || cfg == nil || !cfg.IsConfigured() || (requireEnabled && !cfg.Enabled) {
		return nil, fmt.Errorf("configure and enable image storage in Admin > Backup before sending inline images through Excel BPS")
	}
	// Never use the public CDN or the generated-image expiry for customer inputs.
	cfg.PublicBaseURL = ""
	storage, err := s.settings.factory(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("cannot initialize temporary image storage")
	}
	temporary, ok := storage.(TemporaryImageStorage)
	if !ok {
		return nil, fmt.Errorf("image storage does not support five-minute private links")
	}
	return temporary, nil
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
				ctx, cancel := context.WithTimeout(s.workerCtx, 30*time.Second)
				storage, err := s.resolve(ctx, false)
				if err == nil {
					// Uploads are bounded to 30s. The extra minute ensures no live
					// five-minute signature is deleted while its object is still needed.
					if storage.DeleteExpiredTemporary(ctx, time.Now().Add(-ExcelBPSImageTTL-time.Minute)) != nil {
						logger.L().Warn("excel_bps temporary image cleanup failed; will retry")
					}
				}
				cancel()
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
	s.stopOnce.Do(func() { s.workerCancel(); close(s.stop) })
	// Also supports stopping a service that was constructed but not started.
	s.Start()
	select {
	case <-s.done:
	case <-time.After(5 * time.Second):
	}
}

type excelBPSInlineImage struct {
	placeholder, mime string
	data              []byte
}
type excelBPSImagePlan struct{ images []excelBPSInlineImage }

// Visit actual media parts only. Text, tool schemas and JSON tool arguments
// containing image-like keys must remain opaque and byte-for-byte unchanged.
func excelBPSImageParts(body []byte, visit func(string, gjson.Result) error) error {
	for i, item := range gjson.GetBytes(body, "input").Array() {
		for _, field := range []string{"content", "output"} {
			parts := item.Get(field)
			if !parts.IsArray() {
				continue
			}
			for j, part := range parts.Array() {
				if part.Get("type").String() == "input_image" {
					if err := visit(fmt.Sprintf("input.%d.%s.%d.image_url", i, field, j), part); err != nil {
						return err
					}
				}
			}
		}
	}
	return nil
}

func prepareExcelBPSImages(body []byte) ([]byte, *excelBPSImagePlan, error) {
	plan := &excelBPSImagePlan{}
	seen := make(map[string]bool)
	total, count := 0, 0
	updated := body
	err := excelBPSImageParts(body, func(path string, part gjson.Result) error {
		raw := part.Get("image_url").String()
		if !strings.HasPrefix(strings.ToLower(strings.TrimSpace(raw)), "data:") {
			return nil
		}
		count++
		if count > excelBPSMaxImages {
			return fmt.Errorf("Excel BPS accepts at most 16 inline images per request")
		}
		mime, data, err := decodeExcelBPSImage(raw)
		if err != nil {
			return err
		}
		total += len(data)
		if total > excelBPSMaxTotalImageBytes {
			return fmt.Errorf("Excel BPS inline images exceed 32 MiB in total")
		}
		// Stable placeholders keep turn identities independent of random object
		// keys and expiring signatures. The protocol is validated before upload.
		placeholder := fmt.Sprintf("https://inline-image.invalid/%x", sha256.Sum256(data))
		if !seen[placeholder] {
			plan.images = append(plan.images, excelBPSInlineImage{placeholder, mime, data})
			seen[placeholder] = true
		}
		updated, err = sjson.SetBytes(updated, path, placeholder)
		return err
	})
	return updated, plan, err
}

func decodeExcelBPSImage(raw string) (string, []byte, error) {
	header, encoded, ok := strings.Cut(raw, ",")
	header = strings.ToLower(header)
	if !ok || !strings.HasPrefix(header, "data:image/") || !strings.HasSuffix(header, ";base64") {
		return "", nil, fmt.Errorf("Excel BPS inline images require a base64 PNG, JPEG, GIF or WebP data URL")
	}
	mime := strings.TrimSuffix(strings.TrimPrefix(header, "data:"), ";base64")
	if mime == "image/jpg" {
		mime = "image/jpeg"
	}
	if mime != "image/png" && mime != "image/jpeg" && mime != "image/gif" && mime != "image/webp" {
		return "", nil, fmt.Errorf("Excel BPS supports PNG, JPEG, GIF and WebP images only")
	}
	if len(encoded) > base64.StdEncoding.EncodedLen(excelBPSMaxImageBytes) {
		return "", nil, fmt.Errorf("Excel BPS inline image exceeds 20 MiB")
	}
	data, err := base64.StdEncoding.Strict().DecodeString(encoded)
	if err != nil {
		data, err = base64.RawStdEncoding.Strict().DecodeString(encoded)
	}
	if err != nil || len(data) == 0 || len(data) > excelBPSMaxImageBytes {
		return "", nil, fmt.Errorf("invalid or oversized Excel BPS inline image")
	}
	if http.DetectContentType(data) != mime {
		return "", nil, fmt.Errorf("Excel BPS image content does not match its declared type")
	}
	size, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || size.Width <= 0 || size.Height <= 0 || int64(size.Width)*int64(size.Height) > 40_000_000 {
		return "", nil, fmt.Errorf("invalid Excel BPS image dimensions or more than 40 million pixels")
	}
	return mime, data, nil
}

func (s *ExcelBPSImageService) upload(ctx context.Context, wire []byte, plan *excelBPSImagePlan) ([]byte, error) {
	if plan == nil || len(plan.images) == 0 {
		return wire, nil
	}
	if s == nil || s.resolve == nil {
		return nil, fmt.Errorf("configure and enable image storage in Admin > Backup before sending inline images through Excel BPS")
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	storage, err := s.resolve(ctx, true)
	if err != nil {
		return nil, err
	}
	keys := make([]string, 0, len(plan.images))
	succeeded := false
	defer func() {
		if succeeded {
			return
		}
		cleanup, stop := context.WithTimeout(context.Background(), 5*time.Second)
		defer stop()
		for _, key := range keys {
			_ = storage.DeleteTemporary(cleanup, key)
		}
	}()
	urls := make(map[string]string)
	for _, img := range plan.images {
		key := ExcelBPSImageObjectPrefix + uuid.NewString()
		keys = append(keys, key)
		link, err := storage.SaveTemporary(ctx, key, img.mime, img.data, ExcelBPSImageTTL)
		if err != nil {
			return nil, fmt.Errorf("Excel BPS temporary image upload failed; check image storage configuration and permissions")
		}
		parsed, err := url.Parse(link)
		if err != nil || parsed.Scheme != "https" || parsed.Hostname() == "" || parsed.User != nil {
			return nil, fmt.Errorf("Excel BPS image storage must provide an upstream-accessible HTTPS signed URL")
		}
		urls[img.placeholder] = link
	}
	updated := wire
	err = excelBPSImageParts(wire, func(path string, part gjson.Result) error {
		if link, ok := urls[part.Get("image_url").String()]; ok {
			updated, err = sjson.SetBytes(updated, path, link)
			return err
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("cannot attach Excel BPS temporary images")
	}
	succeeded = true
	return updated, nil
}
