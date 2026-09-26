// Package errorarchive keeps short-lived, encrypted diagnostic captures.
// It never replays requests or stores authentication headers.
package errorarchive

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Config struct {
	Directory    string
	Key          []byte
	Retention    time.Duration
	MaxBytes     int64
	CaptureBytes int
}

type Entry struct {
	ID                string          `json:"id"`
	RequestID         string          `json:"request_id"`
	APIKeyID          int64           `json:"api_key_id"`
	AccountID         int64           `json:"account_id,omitempty"`
	CreatedAt         time.Time       `json:"created_at"`
	ExpiresAt         time.Time       `json:"expires_at"`
	Path              string          `json:"path"`
	Status            int             `json:"status"`
	ContentEncoding   string          `json:"content_encoding"`
	ContentLength     int64           `json:"content_length"`
	ReceivedBytes     int64           `json:"received_bytes"`
	Request           []byte          `json:"request_wire_base64"`
	Truncated         bool            `json:"request_truncated"`
	Complete          bool            `json:"request_complete"`
	CaptureLimited    bool            `json:"capture_limited,omitempty"`
	ReadError         string          `json:"read_error_kind,omitempty"`
	Response          []byte          `json:"response_base64"`
	ResponseTruncated bool            `json:"response_truncated"`
	Diagnostics       json.RawMessage `json:"diagnostics,omitempty"`
}

type Ref struct {
	ID             string    `json:"id,omitempty"`
	State          string    `json:"state"`
	ExpiresAt      time.Time `json:"expires_at,omitzero"`
	Truncated      bool      `json:"request_truncated,omitempty"`
	ReadError      string    `json:"read_error_kind,omitempty"`
	CaptureLimited bool      `json:"capture_limited,omitempty"`
}

type Store struct {
	cfg        Config
	aead       cipher.AEAD
	mu         sync.Mutex
	queue      chan *Entry
	traceSlots chan struct{}
	buffered   atomic.Int64
	active     atomic.Int64
	limited    atomic.Uint64
	done       chan struct{}
	stop       sync.Once
	closed     bool
	fileMu     sync.Mutex
	saved      atomic.Uint64
	dropped    atomic.Uint64
	failed     atomic.Uint64
}

var validID = regexp.MustCompile("^[a-f0-9]{32}$")

func New(cfg Config) (*Store, error) {
	if cfg.Retention <= 0 || cfg.Retention > 30*24*time.Hour || cfg.CaptureBytes <= 0 || cfg.CaptureBytes > 16<<20 || cfg.MaxBytes < int64(cfg.CaptureBytes)*4 || cfg.Directory == "" {
		return nil, errors.New("invalid diagnostic archive limits")
	}
	block, err := aes.NewCipher(cfg.Key)
	if err != nil {
		return nil, err
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	if err = os.MkdirAll(cfg.Directory, 0700); err != nil {
		return nil, err
	}
	info, err := os.Lstat(cfg.Directory)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return nil, errors.New("archive directory must be a real directory")
	}
	if err = os.Chmod(cfg.Directory, 0700); err != nil {
		return nil, err
	}
	s := &Store{cfg: cfg, aead: aead, queue: make(chan *Entry, 8), traceSlots: make(chan struct{}, 8), done: make(chan struct{})}
	if err = s.cleanup(time.Now(), 0); err != nil {
		return nil, err
	}
	go s.run()
	return s, nil
}

func (s *Store) run() {
	defer close(s.done)
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	for {
		select {
		case entry, ok := <-s.queue:
			if !ok {
				return
			}
			if err := s.write(entry); err != nil {
				s.failed.Add(1)
			} else {
				s.saved.Add(1)
			}
			s.buffered.Add(-int64(cap(entry.Request)))
			entry.Request = nil
		case <-ticker.C:
			if err := s.cleanup(time.Now(), 0); err != nil {
				s.failed.Add(1)
			}
		}
	}
}

func (s *Store) Close() {
	if s == nil {
		return
	}
	s.stop.Do(func() { s.mu.Lock(); s.closed = true; close(s.queue); s.mu.Unlock(); <-s.done })
}

func (s *Store) Stats() map[string]any {
	return map[string]any{"saved": s.saved.Load(), "skipped_capacity": s.dropped.Load(), "write_failures": s.failed.Load(), "in_flight": int(s.active.Load()), "queued": len(s.queue), "buffered_bytes": s.buffered.Load(), "buffer_limit_bytes": int64(s.cfg.CaptureBytes) * 8, "memory_limited": s.limited.Load(), "retention_hours": s.cfg.Retention.Hours(), "capture_bytes": s.cfg.CaptureBytes, "max_disk_bytes": s.cfg.MaxBytes}
}

func (s *Store) Capture(body io.ReadCloser) *Capture {
	if s == nil || body == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	// Long, healthy streams must not monopolize eight archive writer slots.
	// Admit a lightweight observer for every request; allocate bytes only as read.
	s.active.Add(1)
	return &Capture{ReadCloser: body, store: s, limit: s.cfg.CaptureBytes}
}

func (s *Store) reserveBytes(n int) bool {
	for {
		used := s.buffered.Load()
		if used+int64(n) > int64(s.cfg.CaptureBytes)*8 {
			return false
		}
		if s.buffered.CompareAndSwap(used, used+int64(n)) {
			return true
		}
	}
}

func (s *Store) write(entry *Entry) error {
	raw, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	nonce := make([]byte, s.aead.NonceSize())
	if _, err = rand.Read(nonce); err != nil {
		return err
	}
	data := s.aead.Seal(nonce, nonce, raw, []byte(entry.ID))
	if int64(len(data)) > s.cfg.MaxBytes {
		return errors.New("archive entry exceeds disk budget")
	}
	s.fileMu.Lock()
	defer s.fileMu.Unlock()
	if err = s.cleanupLocked(time.Now(), int64(len(data))); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(s.cfg.Directory, ".capture-*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if err = tmp.Chmod(0600); err == nil {
		_, err = tmp.Write(data)
	}
	closeErr := tmp.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	return os.Rename(name, filepath.Join(s.cfg.Directory, entry.ID+".enc"))
}

func (s *Store) Read(id string) (*Entry, error) {
	if s == nil || !validID.MatchString(id) {
		return nil, os.ErrNotExist
	}
	s.fileMu.Lock()
	defer s.fileMu.Unlock()
	path := filepath.Join(s.cfg.Directory, id+".enc")
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() > s.cfg.MaxBytes || info.Size() > 64<<20 {
		return nil, os.ErrNotExist
	}
	if time.Since(info.ModTime()) > s.cfg.Retention {
		_ = os.Remove(path)
		return nil, os.ErrNotExist
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	n := s.aead.NonceSize()
	if len(data) < n {
		return nil, errors.New("invalid encrypted archive")
	}
	raw, err := s.aead.Open(nil, data[:n], data[n:], []byte(id))
	if err != nil {
		return nil, errors.New("archive decryption failed")
	}
	var entry Entry
	if err = json.Unmarshal(raw, &entry); err != nil {
		return nil, err
	}
	if entry.ID != id || !time.Now().Before(entry.ExpiresAt) {
		_ = os.Remove(path)
		return nil, os.ErrNotExist
	}
	return &entry, nil
}

func (s *Store) cleanup(now time.Time, reserve int64) error {
	s.fileMu.Lock()
	defer s.fileMu.Unlock()
	return s.cleanupLocked(now, reserve)
}

func (s *Store) cleanupLocked(now time.Time, reserve int64) error {
	entries, err := os.ReadDir(s.cfg.Directory)
	if err != nil {
		return err
	}
	type file struct {
		name string
		size int64
		when time.Time
	}
	var files []file
	var total int64
	for _, entry := range entries {
		name := entry.Name()
		if strings.HasPrefix(name, ".capture-") {
			info, err := entry.Info()
			if err != nil {
				return err
			}
			if info.Mode().IsRegular() && now.Sub(info.ModTime()) >= s.cfg.Retention {
				if err := os.Remove(filepath.Join(s.cfg.Directory, name)); err != nil {
					return err
				}
			}
			continue
		}
		if len(name) != 36 || filepath.Ext(name) != ".enc" || !validID.MatchString(name[:32]) {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			continue
		}
		if now.Sub(info.ModTime()) >= s.cfg.Retention {
			if err = os.Remove(filepath.Join(s.cfg.Directory, name)); err != nil {
				return err
			}
			continue
		}
		files = append(files, file{name, info.Size(), info.ModTime()})
		total += info.Size()
	}
	sort.Slice(files, func(i, j int) bool { return files[i].when.Before(files[j].when) })
	maxFiles := 4096
	if reserve > 0 {
		maxFiles--
	}
	remainingFiles := len(files)
	for _, file := range files {
		if total+reserve <= s.cfg.MaxBytes && remainingFiles <= maxFiles {
			break
		}
		if err = os.Remove(filepath.Join(s.cfg.Directory, file.name)); err != nil {
			return err
		}
		total -= file.size
		remainingFiles--
	}
	return nil
}

type Capture struct {
	io.ReadCloser
	store    *Store
	limit    int
	buf      []byte
	total    int64
	errKind  string
	eof      bool
	released bool
	limited  bool
	trace    *Trace
}

func (c *Capture) Read(p []byte) (int, error) {
	n, err := c.ReadCloser.Read(p)
	if err == io.EOF {
		c.eof = true
	}
	c.total += int64(n)
	if err != nil && err != io.EOF {
		c.OnBodyReadError(err)
	}
	remaining := c.limit - len(c.buf)
	if remaining > n {
		remaining = n
	}
	if remaining > 0 && !c.limited && !c.released {
		need := len(c.buf) + remaining
		if need > cap(c.buf) {
			next := cap(c.buf) * 2
			if next < need {
				next = need
			}
			if next > c.limit {
				next = c.limit
			}
			reserved := c.store.reserveBytes(next - cap(c.buf))
			if !reserved && next != need {
				next = need
				reserved = c.store.reserveBytes(next - cap(c.buf))
			}
			if !reserved {
				c.limited = true
				c.store.limited.Add(1)
				return n, err
			}
			buf := make([]byte, len(c.buf), next)
			copy(buf, c.buf)
			c.buf = buf
		}
		c.buf = append(c.buf, p[:remaining]...)
	}
	return n, err
}

// Release discards captures of successful requests without writing them.
func (c *Capture) Release() {
	if c != nil && !c.released {
		c.released = true
		c.store.buffered.Add(-int64(cap(c.buf)))
		c.buf = nil
		c.store.active.Add(-1)
		c.trace.Release()
	}
}

func (c *Capture) Save(entry *Entry) Ref {
	if c == nil {
		return Ref{State: "capacity_exhausted"}
	}
	if c.released {
		return Ref{State: "unavailable"}
	}
	var id [16]byte
	if _, err := rand.Read(id[:]); err != nil {
		c.Release()
		return Ref{State: "unavailable"}
	}
	entry.ID = hex.EncodeToString(id[:])
	entry.CreatedAt = time.Now()
	entry.ExpiresAt = entry.CreatedAt.Add(c.store.cfg.Retention)
	entry.Request = c.buf
	entry.ReceivedBytes = c.total
	entry.Truncated = c.total > int64(len(c.buf))
	entry.CaptureLimited = c.limited
	entry.ReadError = c.errKind
	entry.Complete = c.errKind == "" && !entry.Truncated && (entry.ContentLength == c.total || (entry.ContentLength < 0 && c.eof))
	ref := Ref{ID: entry.ID, State: "queued", ExpiresAt: entry.ExpiresAt, Truncated: entry.Truncated, ReadError: entry.ReadError, CaptureLimited: c.limited}
	c.store.mu.Lock()
	defer c.store.mu.Unlock()
	if c.store.closed {
		c.Release()
		return Ref{State: "unavailable", Truncated: entry.Truncated, ReadError: entry.ReadError, CaptureLimited: c.limited}
	}
	select {
	case c.store.queue <- entry:
		// Transfer the reserved request buffer to the writer, then release the
		// request observer and diagnostic slot. The writer releases its bytes.
		c.buf = nil
		c.Release()
		return ref
	default:
		c.store.dropped.Add(1)
		c.Release()
		return Ref{State: "queue_full", Truncated: entry.Truncated, ReadError: entry.ReadError, CaptureLimited: c.limited}
	}
}
