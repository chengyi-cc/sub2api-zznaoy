package service

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/klauspost/compress/zstd"
)

// PrepareCodexRequestBody applies the configured request encoding before either
// the core transport or a transport plugin sends the body. GetBody is rebuilt
// to replay the same encoded bytes; an already encoded request is left intact.
func PrepareCodexRequestBody(req *http.Request) error {
	if req == nil || !CodexRequestCompressionEnabled(req.Context()) || req.Body == nil || req.Body == http.NoBody {
		return nil
	}
	if strings.TrimSpace(req.Header.Get("Content-Encoding")) != "" {
		return nil
	}
	raw, err := io.ReadAll(req.Body)
	_ = req.Body.Close()
	if err != nil {
		return fmt.Errorf("read request body for zstd compression: %w", err)
	}
	encoder, err := zstd.NewWriter(nil)
	if err != nil {
		return fmt.Errorf("create zstd encoder: %w", err)
	}
	encoded := encoder.EncodeAll(raw, nil)
	encoder.Close()
	rebuild := func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(encoded)), nil
	}
	req.Body, err = rebuild()
	if err != nil {
		return err
	}
	req.GetBody = rebuild
	req.ContentLength = int64(len(encoded))
	req.TransferEncoding = nil
	if req.Header == nil {
		req.Header = make(http.Header)
	}
	req.Header.Del("Transfer-Encoding")
	req.Header.Set("Content-Encoding", "zstd")
	req.Header.Set("Content-Length", strconv.FormatInt(req.ContentLength, 10))
	return nil
}
