package service

import (
	"bytes"
	"container/list"
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"mime/multipart"
	"net/textproto"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const excelBPSAttachmentsURL = "https://bps.openai.com/basispoints/api/attachments"
const excelBPSAttachmentCacheSize = 1024

type excelBPSAttachmentScope struct {
	account, apiKey int64
	upstreamAccount string
}
type excelBPSAttachmentKey struct {
	scope  excelBPSAttachmentScope
	digest [32]byte
}
type excelBPSAttachmentEntry struct {
	key    excelBPSAttachmentKey
	fileID string
}
type excelBPSAttachmentFlight struct {
	done   chan struct{}
	fileID string
	err    error
}

// Only official file IDs live in this bounded, account/caller-isolated cache.
// It holds neither tokens nor image bytes. Uploads use the selected account's
// actual credentials, proxy and HTTP transport; refreshing a token retains IDs.
type ExcelBPSImageService struct {
	mu                  sync.Mutex
	ids                 map[excelBPSAttachmentKey]*list.Element
	lru                 *list.List
	flights             map[excelBPSAttachmentKey]*excelBPSAttachmentFlight
	slots               chan struct{}
	legacyRoot          string
	startOnce, stopOnce sync.Once
	stop, done          chan struct{}
}

func NewExcelBPSImageService(cfg *config.Config) *ExcelBPSImageService {
	return &ExcelBPSImageService{ids: make(map[excelBPSAttachmentKey]*list.Element), lru: list.New(),
		flights: make(map[excelBPSAttachmentKey]*excelBPSAttachmentFlight), slots: make(chan struct{}, 32),
		legacyRoot: excelBPSLegacyImageRoot(cfg), stop: make(chan struct{}), done: make(chan struct{})}
}

func (s *ExcelBPSImageService) fileID(ctx context.Context, key excelBPSAttachmentKey, upload func() (string, error)) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	s.mu.Lock()
	if e := s.ids[key]; e != nil {
		s.lru.MoveToFront(e)
		id := e.Value.(excelBPSAttachmentEntry).fileID
		s.mu.Unlock()
		return id, nil
	}
	if flight := s.flights[key]; flight != nil {
		s.mu.Unlock()
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-flight.done:
			return flight.fileID, flight.err
		}
	}
	// No unbounded upload queue; identical concurrent pictures still share work.
	select {
	case s.slots <- struct{}{}:
	default:
		s.mu.Unlock()
		return "", fmt.Errorf("too many concurrent image uploads; retry later")
	}
	flight := &excelBPSAttachmentFlight{done: make(chan struct{})}
	s.flights[key] = flight
	s.mu.Unlock()
	id, err := upload()
	s.mu.Lock()
	if err == nil {
		s.ids[key] = s.lru.PushFront(excelBPSAttachmentEntry{key, id})
		for s.lru.Len() > excelBPSAttachmentCacheSize {
			e := s.lru.Back()
			delete(s.ids, e.Value.(excelBPSAttachmentEntry).key)
			s.lru.Remove(e)
		}
	}
	flight.fileID, flight.err = id, err
	delete(s.flights, key)
	close(flight.done)
	<-s.slots
	s.mu.Unlock()
	return id, err
}

func uploadExcelBPSAttachment(ctx context.Context, img excelBPSInlineImage, token, accountID string, account *Account, upstream HTTPUpstream) (string, error) {
	if upstream == nil {
		return "", fmt.Errorf("Excel BPS attachment transport is unavailable")
	}
	ctx, cancel := context.WithTimeout(ctx, 120*time.Second)
	defer cancel()
	ctx = WithHTTPUpstreamRedirectsDisabled(ctx)
	var framing bytes.Buffer
	form := multipart.NewWriter(&framing)
	ext := map[string]string{"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}[img.mime]
	digest := sha256.Sum256(img.data)
	header := textproto.MIMEHeader{}
	header.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename="picture-%x.%s"`, digest[:6], ext))
	header.Set("Content-Type", img.mime)
	if _, err := form.CreatePart(header); err != nil {
		return "", fmt.Errorf("cannot encode image attachment")
	}
	prefixSize := framing.Len()
	if err := form.Close(); err != nil {
		return "", fmt.Errorf("cannot encode image attachment")
	}
	prefix, suffix := framing.Bytes()[:prefixSize], framing.Bytes()[prefixSize:]
	req, err := newExcelBPSRequest(ctx, nil, token, accountID)
	if err != nil {
		return "", fmt.Errorf("cannot build image upload request")
	}
	req.URL, _ = url.Parse(excelBPSAttachmentsURL)
	req.Body = io.NopCloser(io.MultiReader(bytes.NewReader(prefix), bytes.NewReader(img.data), bytes.NewReader(suffix)))
	req.ContentLength = int64(len(prefix) + len(img.data) + len(suffix))
	req.GetBody = nil // No implicit transport replay of uploads.
	req.Header.Set("Content-Type", form.FormDataContentType())
	req.Header.Set("Accept", "application/json")
	proxy := ""
	if account.Proxy != nil {
		proxy = account.Proxy.URL()
	}
	resp, err := upstream.Do(req, proxy, account.ID, account.Concurrency)
	if err != nil {
		return "", fmt.Errorf("Excel BPS attachment upload connection failed; image was not omitted")
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("Excel BPS attachment upload returned HTTP %d; image was not omitted", resp.StatusCode)
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, (64<<10)+1))
	if err != nil || len(raw) > 64<<10 || !gjson.ValidBytes(raw) {
		return "", fmt.Errorf("invalid Excel BPS attachment response")
	}
	id := gjson.GetBytes(raw, "openai_file_id")
	if id.Type != gjson.String || !validExcelBPSFileID(id.String()) {
		return "", fmt.Errorf("Excel BPS attachment response has no valid file ID")
	}
	return id.String(), nil
}

func validExcelBPSFileID(id string) bool {
	if !strings.HasPrefix(id, "file-") || len(id) <= 5 || len(id) > 256 {
		return false
	}
	for _, c := range id[5:] {
		if !(c >= 'a' && c <= 'z') && !(c >= 'A' && c <= 'Z') && !(c >= '0' && c <= '9') && c != '-' && c != '_' {
			return false
		}
	}
	return true
}

func (s *ExcelBPSImageService) upload(ctx context.Context, wire []byte, plan *excelBPSImagePlan, scope excelBPSAttachmentScope, token string, account *Account, upstream HTTPUpstream) ([]byte, error) {
	if plan == nil || len(plan.images) == 0 {
		return wire, nil
	}
	if s == nil {
		return nil, fmt.Errorf("Excel BPS attachment service is unavailable")
	}
	ids := make(map[string]string, len(plan.images))
	for _, img := range plan.images {
		key := excelBPSAttachmentKey{scope, sha256.Sum256(img.data)}
		id, err := s.fileID(ctx, key, func() (string, error) {
			return uploadExcelBPSAttachment(ctx, img, token, scope.upstreamAccount, account, upstream)
		})
		if err != nil {
			return nil, err
		}
		ids[img.placeholder] = id
	}
	updated := wire
	err := excelBPSImageParts(wire, func(path string, part gjson.Result) error {
		id, ok := ids[part.Get("image_url").String()]
		if !ok {
			return nil
		}
		base := strings.TrimSuffix(path, ".image_url")
		var err error
		updated, err = sjson.DeleteBytes(updated, path)
		if err != nil {
			return err
		}
		updated, err = sjson.SetBytes(updated, base+".file_id", id)
		return err
	})
	if err != nil {
		return nil, fmt.Errorf("cannot attach Excel BPS image file IDs")
	}
	return updated, nil
}

// Only explicit invalid-file rejections invalidate cache entries. Never replay
// a model request or silently drop images: the next caller retry reuploads them.
func (s *ExcelBPSImageService) invalidateRejected(scope excelBPSAttachmentScope, wire, raw []byte) {
	if s == nil {
		return
	}
	code := gjson.GetBytes(raw, "error.code").String()
	switch code {
	case "file_not_found", "invalid_file_id", "file_expired":
	default:
		return
	}
	sent := map[string]bool{}
	_ = excelBPSImageParts(wire, func(_ string, part gjson.Result) error {
		if id := part.Get("file_id").String(); id != "" {
			sent[id] = true
		}
		return nil
	})
	s.mu.Lock()
	defer s.mu.Unlock()
	for key, e := range s.ids {
		if key.scope == scope && sent[e.Value.(excelBPSAttachmentEntry).fileID] {
			delete(s.ids, key)
			s.lru.Remove(e)
		}
	}
}
