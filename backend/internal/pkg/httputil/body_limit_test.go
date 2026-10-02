package httputil

import (
	"bytes"
	"compress/gzip"
	"compress/zlib"
	"errors"
	"github.com/klauspost/compress/zstd"
	"net/http"
	"strings"
	"testing"
)

func encodeBodyLimitFixture(t *testing.T, encoding string, body []byte) []byte {
	t.Helper()
	var b bytes.Buffer
	switch encoding {
	case "gzip":
		w := gzip.NewWriter(&b)
		_, err := w.Write(body)
		if err != nil {
			t.Fatal(err)
		}
		if err = w.Close(); err != nil {
			t.Fatal(err)
		}
	case "deflate":
		w := zlib.NewWriter(&b)
		_, err := w.Write(body)
		if err != nil {
			t.Fatal(err)
		}
		if err = w.Close(); err != nil {
			t.Fatal(err)
		}
	case "zstd":
		w, err := zstd.NewWriter(nil)
		if err != nil {
			t.Fatal(err)
		}
		defer w.Close()
		return w.EncodeAll(body, nil)
	}
	return b.Bytes()
}

func TestDecodedBodyExactLimitAndOneByteOver(t *testing.T) {
	for _, enc := range []string{"gzip", "deflate", "zstd"} {
		t.Run(enc, func(t *testing.T) {
			for _, n := range []int{4095, 4096, 4097} {
				data := bytes.Repeat([]byte("x"), n)
				wire := encodeBodyLimitFixture(t, enc, data)
				got, err := decompressRequestBodyWithLimit(enc, wire, 4096)
				if n > 4096 {
					var large *http.MaxBytesError
					if !errors.As(err, &large) || large.Limit != 4096 || got != nil {
						t.Fatalf("over-limit not rejected: %v bytes=%d", err, len(got))
					}
				} else if err != nil || !bytes.Equal(got, data) {
					t.Fatalf("valid boundary rejected: %v", err)
				}
			}
		})
	}
}

func TestDecodedBodyHonorsRouteAndHandlerBudget(t *testing.T) {
	data := []byte("{\"input\":\"" + strings.Repeat("x", 1500) + "\"}")
	for _, limits := range [][2]int64{{1024, 4096}, {4096, 1024}, {4096, 4096}} {
		req := newRequestWithBody(t, encodeBodyLimitFixture(t, "gzip", data), "gzip")
		req = req.WithContext(WithRequestBodyLimit(req.Context(), limits[0]))
		got, err := ReadLenientJSONRequestBodyWithPrealloc(req, limits[1])
		limit := min(limits[0], limits[1])
		if limit < 1500 {
			var large *http.MaxBytesError
			if !errors.As(err, &large) || large.Limit != limit || got != nil {
				t.Fatalf("wrong effective limit: %v", err)
			}
			if req.Header.Get("Content-Encoding") != "gzip" {
				t.Fatal("failed decoding changed headers")
			}
		} else if err != nil || !bytes.Equal(got, data) {
			t.Fatalf("allowed body rejected: %v", err)
		}
	}
}

func TestDecodedBodyReadsTrailerAtExactLimit(t *testing.T) {
	data := bytes.Repeat([]byte("x"), 4096)
	for _, enc := range []string{"gzip", "deflate", "zstd"} {
		wire := encodeBodyLimitFixture(t, enc, data)
		wire = wire[:len(wire)-2]
		if got, err := decompressRequestBodyWithLimit(enc, wire, int64(len(data))); err == nil || got != nil {
			t.Fatalf("%s accepted truncated compressed frame", enc)
		}
	}
}

func TestDecodedBodyNestedLimitsAndMaxIntDoNotWidenOrOverflow(t *testing.T) {
	req := newRequestWithBody(t, encodeBodyLimitFixture(t, "gzip", []byte(samplePayload)), "gzip")
	req = req.WithContext(WithRequestBodyLimit(WithRequestBodyLimit(req.Context(), 4096), 8192))
	if requestBodySizeLimit(req, 16384) != 4096 {
		t.Fatal("nested middleware widened a route budget")
	}
	req = newRequestWithBody(t, encodeBodyLimitFixture(t, "gzip", []byte(samplePayload)), "gzip")
	req = req.WithContext(WithRequestBodyLimit(req.Context(), int64(^uint64(0)>>1)))
	got, err := ReadRequestBodyWithPrealloc(req)
	if err != nil || string(got) != samplePayload {
		t.Fatalf("overflow truncated body: %v", err)
	}
}

func TestConfiguredDecodedBudgetCanExceedLegacy64MiB(t *testing.T) {
	if testing.Short() {
		t.Skip("large compressed-body boundary")
	}
	data := bytes.Repeat([]byte("x"), (64<<20)+16)
	wire := encodeBodyLimitFixture(t, "gzip", data)
	req := newRequestWithBody(t, wire, "gzip")
	req = req.WithContext(WithRequestBodyLimit(req.Context(), 70<<20))
	got, err := ReadRequestBodyWithPrealloc(req)
	if err != nil || !bytes.Equal(got, data) {
		t.Fatalf("configured budget still truncated: bytes=%d error=%v", len(got), err)
	}
}
