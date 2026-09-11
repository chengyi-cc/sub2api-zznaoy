package service

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	pluginv1 "github.com/Wei-Shaw/sub2api/pkg/pluginapi/v1"
	"github.com/klauspost/compress/zstd"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
)

type compressionPluginCapture struct {
	start *pluginv1.ForwardRequestStart
	body  []byte
}

type compressionPluginServer struct {
	pluginv1.UnimplementedTransportPluginServer
	captured chan compressionPluginCapture
}

func (s *compressionPluginServer) Forward(stream grpc.BidiStreamingServer[pluginv1.ForwardRequest, pluginv1.ForwardResponse]) error {
	var captured compressionPluginCapture
	for {
		frame, err := stream.Recv()
		if err != nil {
			return err
		}
		if start := frame.GetStart(); start != nil {
			captured.start = start
		}
		captured.body = append(captured.body, frame.GetBodyChunk()...)
		if frame.GetBodyEnd() {
			break
		}
	}
	s.captured <- captured
	if err := stream.Send(&pluginv1.ForwardResponse{Frame: &pluginv1.ForwardResponse_Start{Start: &pluginv1.ForwardResponseStart{StatusCode: 200}}}); err != nil {
		return err
	}
	return stream.Send(&pluginv1.ForwardResponse{Frame: &pluginv1.ForwardResponse_End{End: &pluginv1.ForwardResponseEnd{}}})
}

func TestPluginRuntimeRequestCompression(t *testing.T) {
	for _, tc := range []struct {
		name, encoding string
		marked         bool
	}{
		{name: "enabled", marked: true},
		{name: "disabled"},
		{name: "already encoded", marked: true, encoding: "br"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			listener := bufconn.Listen(1024 * 1024)
			server := grpc.NewServer()
			capture := &compressionPluginServer{captured: make(chan compressionPluginCapture, 1)}
			pluginv1.RegisterTransportPluginServer(server, capture)
			go func() { _ = server.Serve(listener) }()
			t.Cleanup(server.Stop)
			t.Cleanup(func() { _ = listener.Close() })
			conn, err := grpc.NewClient("passthrough:///compression-test",
				grpc.WithTransportCredentials(insecure.NewCredentials()),
				grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return listener.DialContext(ctx) }))
			require.NoError(t, err)
			t.Cleanup(func() { _ = conn.Close() })
			runtime := &pluginRuntime{api: pluginv1.NewTransportPluginClient(conn)}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if tc.marked {
				ctx = WithCodexRequestCompression(ctx)
			}
			original := []byte(`{"model":"gpt-5.4","input":"hello"}`)
			request, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://example.com/v1/responses", bytes.NewReader(original))
			require.NoError(t, err)
			if tc.encoding != "" {
				request.Header.Set("Content-Encoding", tc.encoding)
			}
			require.True(t, runtime.beginRequest())
			response, err := runtime.roundTrip(ctx, request, "", &Account{ID: 1, Platform: PlatformOpenAI, Type: AccountTypeOAuth})
			if err != nil {
				runtime.finishRequest()
			}
			require.NoError(t, err)
			_, err = io.ReadAll(response.Body)
			require.NoError(t, err)
			require.NoError(t, response.Body.Close())
			require.Zero(t, runtime.inFlight.Load())
			captured := <-capture.captured
			require.NotNil(t, captured.start)
			require.Equal(t, int64(len(captured.body)), captured.start.ContentLength)
			encoding := headersFromPlugin(captured.start.Headers).Get("Content-Encoding")
			if tc.marked && tc.encoding == "" {
				require.Equal(t, "zstd", encoding)
				decoder, err := zstd.NewReader(nil)
				require.NoError(t, err)
				defer decoder.Close()
				decoded, err := decoder.DecodeAll(captured.body, nil)
				require.NoError(t, err)
				require.Equal(t, original, decoded)
			} else {
				require.Equal(t, tc.encoding, encoding)
				require.Equal(t, original, captured.body)
			}
			replay, err := request.GetBody()
			require.NoError(t, err)
			defer replay.Close()
			replayed, err := io.ReadAll(replay)
			require.NoError(t, err)
			require.Equal(t, captured.body, replayed)
		})
	}
}

func TestOpenAIRequestCompressionConfiguration(t *testing.T) {
	for _, tc := range []struct {
		name, accountType string
		disabled, want    bool
	}{
		{name: "oauth default", accountType: AccountTypeOAuth, want: true},
		{name: "oauth disabled", accountType: AccountTypeOAuth, disabled: true},
		{name: "setup token", accountType: AccountTypeSetupToken, want: true},
		{name: "api key", accountType: AccountTypeAPIKey},
	} {
		for _, probe := range []bool{false, true} {
			name := tc.name + "/forward"
			if probe {
				name = tc.name + "/probe"
			}
			t.Run(name, func(t *testing.T) {
				cfg := &config.Config{}
				cfg.Gateway.DisableCodexZstdRequestBody = tc.disabled
				upstream := &httpUpstreamRecorder{err: errors.New("capture")}
				account := &Account{ID: 1, Platform: PlatformOpenAI, Type: tc.accountType}
				request, err := http.NewRequest(http.MethodPost, "https://example.com/v1/responses", bytes.NewBufferString(`{"input":"hello"}`))
				require.NoError(t, err)
				if probe {
					_, err = (&AccountTestService{cfg: cfg, httpUpstream: upstream}).doOpenAIAccountTestUpstream(request, "", account, false)
				} else {
					_, err = (&OpenAIGatewayService{cfg: cfg, httpUpstream: upstream}).doOpenAIUpstream(request, "", account)
				}
				require.Error(t, err)
				require.NotNil(t, upstream.lastReq)
				require.Equal(t, tc.want, CodexRequestCompressionEnabled(upstream.lastReq.Context()))
			})
		}
	}
}

type compressionFailingBody struct{ closed bool }

func (*compressionFailingBody) Read([]byte) (int, error) {
	return 0, errors.New("request read failed")
}
func (b *compressionFailingBody) Close() error { b.closed = true; return nil }

func TestPrepareCodexRequestBodyEmptyAndReadFailure(t *testing.T) {
	ctx := WithCodexRequestCompression(context.Background())
	empty, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://example.com", bytes.NewReader(nil))
	require.NoError(t, err)
	require.NoError(t, PrepareCodexRequestBody(empty))
	require.Empty(t, empty.Header.Get("Content-Encoding"))
	require.Equal(t, http.NoBody, empty.Body)

	failing := &compressionFailingBody{}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://example.com", failing)
	require.NoError(t, err)
	// A failed read must return before opening a plugin stream. No plugin API
	// is configured here; reaching it would panic instead of returning the error.
	_, err = (&pluginRuntime{}).roundTrip(ctx, request, "", &Account{ID: 1})
	require.ErrorContains(t, err, "request read failed")
	require.True(t, failing.closed)
	require.Empty(t, request.Header.Get("Content-Encoding"))
}

func TestPrepareCodexRequestBodyNormalizesFraming(t *testing.T) {
	request, err := http.NewRequestWithContext(WithCodexRequestCompression(context.Background()), http.MethodPost, "https://example.com", bytes.NewBufferString("payload"))
	require.NoError(t, err)
	request.TransferEncoding = []string{"chunked"}
	request.Header.Set("Transfer-Encoding", "chunked")
	require.NoError(t, PrepareCodexRequestBody(request))
	require.Empty(t, request.TransferEncoding)
	require.Empty(t, request.Header.Get("Transfer-Encoding"))
	encoded, err := io.ReadAll(request.Body)
	require.NoError(t, err)
	require.Equal(t, int64(len(encoded)), request.ContentLength)

	// Repeated preparation must preserve the replay factory and encoded bytes.
	require.NoError(t, PrepareCodexRequestBody(request))
	replay, err := request.GetBody()
	require.NoError(t, err)
	defer replay.Close()
	replayed, err := io.ReadAll(replay)
	require.NoError(t, err)
	require.Equal(t, encoded, replayed)
}
