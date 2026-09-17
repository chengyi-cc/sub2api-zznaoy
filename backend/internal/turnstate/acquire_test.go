package turnstate

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestAcquireThroughAuthenticatedTLSProxyAndAlwaysRetiresLease(test *testing.T) {
	private, err := rsa.GenerateKey(rand.Reader, 2048)
	require.NoError(test, err)
	template := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "chatgpt.com"}, DNSNames: []string{"chatgpt.com"}, NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour), KeyUsage: x509.KeyUsageDigitalSignature, ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &private.PublicKey, private)
	require.NoError(test, err)
	cert, err := tls.X509KeyPair(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(private)}))
	require.NoError(test, err)
	normal := stateValue(time.Now(), 217)
	var abnormal atomic.Bool
	var upstreamCalls atomic.Int64
	upstream := httptest.NewUnstartedServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		upstreamCalls.Add(1)
		require.Equal(test, "/backend-api/codex/responses", request.URL.Path)
		require.Equal(test, "Bearer account-secret", request.Header.Get("Authorization"))
		require.Empty(test, request.Header.Get(Header))
		require.Empty(test, request.Header.Get("Cookie"))
		require.NotEqual(test, "old-session", request.Header.Get("Session-Id"))
		var payload map[string]any
		require.NoError(test, json.NewDecoder(request.Body).Decode(&payload))
		require.Equal(test, "model-a", payload["model"])
		require.Equal(test, false, payload["store"])
		require.Equal(test, true, payload["stream"])
		value := normal
		if abnormal.Load() {
			value = stateValue(time.Now(), 233)
		}
		writer.Header().Set(Header, value)
		_, _ = io.WriteString(writer, "data: {}\n\n")
	}))
	upstream.TLS = &tls.Config{Certificates: []tls.Certificate{cert}, MinVersion: tls.VersionTLS12}
	upstream.StartTLS()
	defer upstream.Close()
	var allocated atomic.Int64
	var released atomic.Int64
	var proxy *httptest.Server
	leaseID := strings.Repeat("a", 32)
	proxy = httptest.NewTLSServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.Method == http.MethodConnect {
			require.Equal(test, "chatgpt.com:443", request.Host)
			require.NotEmpty(test, request.Header.Get("Proxy-Authorization"))
			target, connectErr := net.DialTimeout("tcp", upstream.Listener.Addr().String(), time.Second)
			require.NoError(test, connectErr)
			defer func() { _ = target.Close() }()
			hijacker, ok := writer.(http.Hijacker)
			require.True(test, ok)
			connection, buffered, hijackErr := hijacker.Hijack()
			require.NoError(test, hijackErr)
			defer func() { _ = connection.Close() }()
			_, _ = connection.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n"))
			copied := make(chan struct{})
			go func() { _, _ = io.Copy(target, buffered); _ = target.Close(); close(copied) }()
			_, _ = io.Copy(connection, target)
			_ = connection.Close()
			<-copied
			return
		}
		require.Equal(test, "Bearer "+strings.Repeat("a", 48), request.Header.Get("Authorization"))
		if request.Method == http.MethodPost && request.URL.Path == "/v1/leases" {
			allocated.Add(1)
			endpoint, _ := url.Parse(proxy.URL)
			endpoint.User = url.UserPassword(leaseID, "proxy-secret")
			_ = json.NewEncoder(writer).Encode(lease{ID: leaseID, ProxyURL: endpoint.String(), IPv6: "2606:4700::1234"})
		} else if request.Method == http.MethodDelete && request.URL.Path == "/v1/leases/"+leaseID {
			released.Add(1)
			_, _ = io.WriteString(writer, "{}")
		} else {
			writer.WriteHeader(http.StatusNotFound)
		}
	}))
	defer proxy.Close()
	manager, _ := managerForTest(test)
	manager.config.URL = proxy.URL
	manager.tlsConfig.RootCAs.AddCert(proxy.Certificate())
	leaf, err := x509.ParseCertificate(der)
	require.NoError(test, err)
	manager.tlsConfig.RootCAs.AddCert(leaf)
	headers := make(http.Header)
	headers.Set("Authorization", "Bearer account-secret")
	headers.Set(Header, "old-state")
	headers.Set("Cookie", "old-cookie")
	headers.Set("Session-Id", "old-session")
	record, err := manager.acquire(context.Background(), headers, "model-a")
	require.NoError(test, err)
	require.Equal(test, normal, record.Value)
	require.EqualValues(test, 1, released.Load())
	abnormal.Store(true)
	_, err = manager.acquire(context.Background(), headers, "model-a")
	require.ErrorContains(test, err, "312")
	require.EqualValues(test, 2, released.Load())
	require.EqualValues(test, 2, allocated.Load())
	require.EqualValues(test, 2, upstreamCalls.Load())
}
