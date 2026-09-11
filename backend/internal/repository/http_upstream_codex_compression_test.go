package repository

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/klauspost/compress/zstd"
	"github.com/stretchr/testify/require"
)

func TestPrepareCodexRequestBodyCompressesAndReplaysIdentically(t *testing.T) {
	original := []byte(`{"model":"gpt-5.4","input":"hello"}`)
	request, err := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/responses", bytes.NewReader(original))
	require.NoError(t, err)
	request = request.WithContext(service.WithCodexRequestCompression(request.Context()))

	require.NoError(t, service.PrepareCodexRequestBody(request))
	require.Equal(t, "zstd", request.Header.Get("Content-Encoding"))
	require.Equal(t, strconv.FormatInt(request.ContentLength, 10), request.Header.Get("Content-Length"))

	encoded, err := io.ReadAll(request.Body)
	require.NoError(t, err)
	decoder, err := zstd.NewReader(nil)
	require.NoError(t, err)
	require.Equal(t, original, mustDecodeZstd(t, decoder, encoded))
	decoder.Close()

	replayed, err := request.GetBody()
	require.NoError(t, err)
	replayedBytes, err := io.ReadAll(replayed)
	require.NoError(t, err)
	require.Equal(t, encoded, replayedBytes)
}

func TestHTTPUpstreamRequestCompressionOnWire(t *testing.T) {
	for _, tlsEntry := range []bool{false, true} {
		for _, marked := range []bool{false, true} {
			name := "normal/"
			if tlsEntry {
				name = "tls-entry/"
			}
			name += strconv.FormatBool(marked)
			t.Run(name, func(t *testing.T) {
				type capture struct {
					body     []byte
					encoding string
					length   int64
					transfer []string
					readErr  error
				}
				captured := make(chan capture, 1)
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					body, readErr := io.ReadAll(r.Body)
					captured <- capture{body, r.Header.Get("Content-Encoding"), r.ContentLength, r.TransferEncoding, readErr}
					w.WriteHeader(http.StatusNoContent)
				}))
				defer server.Close()
				original := []byte(`{"input":"payload"}`)
				request, err := http.NewRequest(http.MethodPost, server.URL, bytes.NewReader(original))
				require.NoError(t, err)
				if marked {
					request = request.WithContext(service.WithCodexRequestCompression(request.Context()))
				}
				upstream := NewHTTPUpstream(nil)
				var response *http.Response
				if tlsEntry {
					response, err = upstream.DoWithTLS(request, "", 1, 1, nil)
				} else {
					response, err = upstream.Do(request, "", 1, 1)
				}
				require.NoError(t, err)
				require.NoError(t, response.Body.Close())
				wire := <-captured
				require.NoError(t, wire.readErr)
				require.Equal(t, int64(len(wire.body)), wire.length)
				require.Empty(t, wire.transfer)
				if marked {
					require.Equal(t, "zstd", wire.encoding)
					decoder, err := zstd.NewReader(nil)
					require.NoError(t, err)
					defer decoder.Close()
					require.Equal(t, original, mustDecodeZstd(t, decoder, wire.body))
				} else {
					require.Empty(t, wire.encoding)
					require.Equal(t, original, wire.body)
				}
			})
		}
	}
}

func TestPrepareCodexRequestBodySkipsUnmarkedAndAlreadyEncodedRequests(t *testing.T) {
	original := []byte(`{"input":"hello"}`)
	request, err := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/responses", bytes.NewReader(original))
	require.NoError(t, err)
	require.NoError(t, service.PrepareCodexRequestBody(request))
	require.Empty(t, request.Header.Get("Content-Encoding"))

	request = request.WithContext(service.WithCodexRequestCompression(request.Context()))
	request.Header.Set("Content-Encoding", "br")
	require.NoError(t, service.PrepareCodexRequestBody(request))
	require.Equal(t, "br", request.Header.Get("Content-Encoding"))
}

func mustDecodeZstd(t *testing.T, decoder *zstd.Decoder, encoded []byte) []byte {
	t.Helper()
	decoded, err := decoder.DecodeAll(encoded, nil)
	require.NoError(t, err)
	return decoded
}
