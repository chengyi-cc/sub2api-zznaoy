package service

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	coderws "github.com/coder/websocket"
	"github.com/stretchr/testify/require"
)

// Exercise an actual TLS proxy + CONNECT tunnel + secure WebSocket upgrade.
// Trust is injected only in this test; production uses system certificate roots.
func TestOpenAIWSTLS_HTTPSProxy(t *testing.T) {
	upstream := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := coderws.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer conn.CloseNow()
		_ = conn.Write(r.Context(), coderws.MessageText, []byte("connected"))
		_ = conn.Close(coderws.StatusNormalClosure, "done")
	}))
	defer upstream.Close()
	var connectCalls atomic.Int32
	proxy := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodConnect || r.Host != upstream.Listener.Addr().String() {
			http.Error(w, "unexpected target", 400)
			return
		}
		connectCalls.Add(1)
		dst, err := net.DialTimeout("tcp", r.Host, time.Second)
		if err != nil {
			http.Error(w, "dial failed", 502)
			return
		}
		defer dst.Close()
		conn, brw, err := w.(http.Hijacker).Hijack()
		if err != nil {
			return
		}
		defer conn.Close()
		_, _ = brw.WriteString("HTTP/1.1 200 Connection Established\r\n\r\n")
		_ = brw.Flush()
		done := make(chan struct{})
		go func() { _, _ = io.Copy(dst, brw); _ = dst.Close(); close(done) }()
		_, _ = io.Copy(conn, dst)
		_ = conn.Close()
		<-done
	}))
	defer proxy.Close()
	profile := &tlsfingerprint.Profile{Name: "test"}
	client, err := newOpenAIWSTLSHTTPClient(profile, proxy.URL)
	require.NoError(t, err)
	defer client.CloseIdleConnections()
	transport := client.Transport.(*http.Transport)
	require.Nil(t, transport.DialTLSContext, "HTTPS proxy uses the standard certificate-verifying transport")
	require.Equal(t, 10*time.Second, transport.TLSHandshakeTimeout)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	wsURL := "wss" + strings.TrimPrefix(upstream.URL, "https")
	_, _, err = coderws.Dial(ctx, wsURL, &coderws.DialOptions{HTTPClient: client})
	require.Error(t, err, "an untrusted proxy must be rejected")
	var verificationError *tls.CertificateVerificationError
	require.ErrorAs(t, err, &verificationError)
	require.Zero(t, connectCalls.Load(), "CONNECT must not be sent before proxy verification")

	// httptest's proxy and upstream use the same local test certificate.
	roots := x509.NewCertPool()
	roots.AddCert(proxy.Certificate())
	transport.TLSClientConfig = &tls.Config{RootCAs: roots, MinVersion: tls.VersionTLS12}
	conn, _, err := coderws.Dial(ctx, wsURL, &coderws.DialOptions{HTTPClient: client})
	require.NoError(t, err)
	defer conn.CloseNow()
	_, message, err := conn.Read(ctx)
	require.NoError(t, err)
	require.Equal(t, "connected", string(message))
	require.EqualValues(t, 1, connectCalls.Load())
}

func TestOpenAIWSTLS_DirectRejectsUntrustedCertificate(t *testing.T) {
	upstream := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("request must not reach an untrusted upstream")
	}))
	defer upstream.Close()
	client, err := newOpenAIWSTLSHTTPClient(&tlsfingerprint.Profile{Name: "test"}, "")
	require.NoError(t, err)
	defer client.CloseIdleConnections()
	client.Timeout = 5 * time.Second
	_, err = client.Get(upstream.URL)
	require.Error(t, err)
	var unknownAuthority x509.UnknownAuthorityError
	require.ErrorAs(t, err, &unknownAuthority)
}
