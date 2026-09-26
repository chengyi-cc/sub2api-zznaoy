package errorarchive

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func testConfig(t *testing.T) Config {
	return Config{Directory: t.TempDir(), Key: bytes.Repeat([]byte{7}, 32), Retention: time.Hour, MaxBytes: 1 << 20, CaptureBytes: 1024}
}
func TestArchiveEncryptedRoundTripExpiryAndIsolation(t *testing.T) {
	cfg := testConfig(t)
	s, err := New(cfg)
	require.NoError(t, err)
	c := s.Capture(io.NopCloser(strings.NewReader("private conversation")))
	got, err := io.ReadAll(c)
	require.NoError(t, err)
	require.Equal(t, "private conversation", string(got))
	ref := c.Save(&Entry{RequestID: "request-1", APIKeyID: 12, ContentLength: int64(len(got)), Status: 502})
	s.Close()
	require.Equal(t, uint64(1), s.saved.Load())
	data, err := os.ReadFile(filepath.Join(cfg.Directory, ref.ID+".enc"))
	require.NoError(t, err)
	require.NotContains(t, string(data), "private conversation")
	require.False(t, json.Valid(data), "the on-disk archive must not be plaintext JSON")
	entry, err := s.Read(ref.ID)
	require.NoError(t, err)
	require.Equal(t, got, entry.Request)
	require.True(t, entry.Complete)
	require.Equal(t, int64(12), entry.APIKeyID)
	tampered := append([]byte(nil), data...)
	tampered[len(tampered)-1] ^= 1
	require.NoError(t, os.WriteFile(filepath.Join(cfg.Directory, ref.ID+".enc"), tampered, 0600))
	_, err = s.Read(ref.ID)
	require.ErrorContains(t, err, "decryption")
	require.NoError(t, os.WriteFile(filepath.Join(cfg.Directory, ref.ID+".enc"), data, 0600))
	_, err = s.Read("../outside")
	require.Error(t, err)
	cfg.Key = bytes.Repeat([]byte{9}, 32)
	other, err := New(cfg)
	require.NoError(t, err)
	defer other.Close()
	_, err = other.Read(ref.ID)
	require.ErrorContains(t, err, "decryption")
	past := time.Now().Add(-2 * time.Hour)
	require.NoError(t, os.Chtimes(filepath.Join(cfg.Directory, ref.ID+".enc"), past, past))
	_, err = s.Read(ref.ID)
	require.ErrorIs(t, err, os.ErrNotExist)
}
func TestArchiveCaptureBoundsAndSuccessDiscard(t *testing.T) {
	cfg := testConfig(t)
	cfg.CaptureBytes = 4
	s, err := New(cfg)
	require.NoError(t, err)
	success := s.Capture(io.NopCloser(strings.NewReader("success")))
	_, err = io.ReadAll(success)
	require.NoError(t, err)
	success.Release()
	c := s.Capture(io.NopCloser(strings.NewReader("123456")))
	_, err = io.ReadAll(c)
	require.NoError(t, err)
	c.OnBodyReadError(errors.New(`decode Content-Encoding "gzip": secret-body`))
	ref := c.Save(&Entry{ContentLength: 6})
	s.Close()
	entry, err := s.Read(ref.ID)
	require.NoError(t, err)
	require.Equal(t, []byte("1234"), entry.Request)
	require.True(t, entry.Truncated)
	require.False(t, entry.Complete)
	require.Equal(t, int64(6), entry.ReceivedBytes)
	require.Equal(t, "decode_content_encoding", entry.ReadError)
	files, err := os.ReadDir(cfg.Directory)
	require.NoError(t, err)
	require.Len(t, files, 1)
}
func TestArchiveCapacityNeverBlocksAndShutdown(t *testing.T) {
	s, err := New(testConfig(t))
	require.NoError(t, err)
	var captures []*Capture
	for i := 0; i < 8; i++ {
		c := s.Capture(io.NopCloser(strings.NewReader("body")))
		require.NotNil(t, c)
		captures = append(captures, c)
	}
	require.Nil(t, s.Capture(io.NopCloser(strings.NewReader("overflow"))))
	require.Equal(t, uint64(1), s.dropped.Load())
	s.Close()
	for _, c := range captures {
		require.Equal(t, "unavailable", c.Save(&Entry{}).State)
		c.Release()
	}
	require.Empty(t, s.slots)
	require.Nil(t, s.Capture(io.NopCloser(strings.NewReader("closed"))))
}
func TestArchiveDiskCapAndIdleCleanup(t *testing.T) {
	cfg := testConfig(t)
	cfg.MaxBytes = 4096
	s, err := New(cfg)
	require.NoError(t, err)
	defer s.Close()
	for i := 0; i < 9; i++ {
		entry := &Entry{ID: strings.Repeat("0", 31) + string(rune('1'+i)), CreatedAt: time.Now(), ExpiresAt: time.Now().Add(time.Hour), Request: bytes.Repeat([]byte("x"), 1024)}
		require.NoError(t, s.write(entry))
	}
	files, err := os.ReadDir(cfg.Directory)
	require.NoError(t, err)
	require.Less(t, len(files), 9)
	var size int64
	for _, f := range files {
		info, err := f.Info()
		require.NoError(t, err)
		size += info.Size()
	}
	require.LessOrEqual(t, size, cfg.MaxBytes)
	require.NoError(t, s.cleanup(time.Now().Add(2*time.Hour), 0))
	files, err = os.ReadDir(cfg.Directory)
	require.NoError(t, err)
	require.Empty(t, files)
}
func TestDiagnosticTraceBoundsAndAbsentNoop(t *testing.T) {
	AddDiagnostic(context.Background(), "ignored", nil, nil)
	ctx, trace := WithTrace(context.Background())
	for i := 0; i < 10; i++ {
		AddDiagnostic(ctx, "invalid", bytes.Repeat([]byte("q"), 2<<20), bytes.Repeat([]byte("r"), 2<<20))
	}
	require.Len(t, trace.items, 2)
	require.Zero(t, trace.remaining)
	require.True(t, trace.items[0].Truncated)
	require.LessOrEqual(t, len(trace.Snapshot()), 3<<20)
}

func TestArchiveUnknownLengthPartialReadIsNotComplete(t *testing.T) {
	s, err := New(testConfig(t))
	require.NoError(t, err)
	c := s.Capture(io.NopCloser(strings.NewReader("partial upload")))
	_, err = c.Read(make([]byte, 3))
	require.NoError(t, err)
	ref := c.Save(&Entry{ContentLength: -1})
	s.Close()
	entry, err := s.Read(ref.ID)
	require.NoError(t, err)
	require.False(t, entry.Complete)
	require.Equal(t, []byte("par"), entry.Request)
}
