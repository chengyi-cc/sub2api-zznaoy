package turnstate

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type testContextDialer func(context.Context, string, string) (net.Conn, error)

func (dialer testContextDialer) DialContext(ctx context.Context, network, address string) (net.Conn, error) {
	return dialer(ctx, network, address)
}

func TestPurchasedProbeVerifiesCountryIPv4AndReusesConnectionWithoutReadingGeneration(test *testing.T) {
	private, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(test, err)
	template := &x509.Certificate{SerialNumber: big.NewInt(1), DNSNames: []string{"chatgpt.com", "cloudflare-dns.com"}, NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour), KeyUsage: x509.KeyUsageDigitalSignature, ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &private.PublicKey, private)
	require.NoError(test, err)
	leaf, err := x509.ParseCertificate(der)
	require.NoError(test, err)
	var trace atomic.Value
	trace.Store("ip=198.51.100.15\nloc=US\n")
	var tracePeer atomic.Value
	var calls atomic.Int64
	var dnsCalls atomic.Int64
	value := stateValue(time.Now(), 249)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/dns-query":
			dnsCalls.Add(1)
			require.Empty(test, request.Header.Get("Authorization"))
			_, _ = io.WriteString(writer, `{"Status":0,"Answer":[{"type":1,"TTL":60,"data":"203.0.113.10"}]}`)
		case "/cdn-cgi/trace":
			require.Empty(test, request.Header.Get("Authorization"))
			tracePeer.Store(request.RemoteAddr)
			_, _ = io.WriteString(writer, trace.Load().(string))
		case "/backend-api/codex/responses":
			calls.Add(1)
			require.Equal(test, tracePeer.Load().(string), request.RemoteAddr)
			require.Equal(test, "Bearer account-secret", request.Header.Get("Authorization"))
			require.Empty(test, request.Header.Get(Header))
			require.Empty(test, request.Header.Get("Cookie"))
			require.Empty(test, request.Header.Get("Proxy-Authorization"))
			require.Len(test, request.Header.Get("Session-Id"), 16)
			var payload map[string]any
			require.NoError(test, json.NewDecoder(request.Body).Decode(&payload))
			require.Equal(test, "model-a", payload["model"])
			writer.Header().Set(Header, value)
			writer.Header().Set("Content-Type", "text/event-stream")
			writer.WriteHeader(200)
			writer.(http.Flusher).Flush()
			<-request.Context().Done()
		default:
			writer.WriteHeader(404)
		}
	}))
	server.TLS = &tls.Config{Certificates: []tls.Certificate{{Certificate: [][]byte{der}, PrivateKey: private}}, MinVersion: tls.VersionTLS12}
	server.StartTLS()
	defer server.Close()
	manager, _ := managerForTest(test)
	manager.tlsConfig.RootCAs.AddCert(leaf)
	manager.dialPurchased = func(country string) (contextDialer, error) {
		require.Equal(test, "US", country)
		return testContextDialer(func(ctx context.Context, network, address string) (net.Conn, error) {
			if address != "cloudflare-dns.com:443" && address != "203.0.113.10:443" {
				return nil, errors.New("unexpected target")
			}
			return (&net.Dialer{}).DialContext(ctx, network, server.Listener.Addr().String())
		}), nil
	}
	ctx := context.WithValue(context.Background(), acquisitionKey{}, acquisitionOptions{Options: Options{}.Normalized(), Country: "US"})
	headers := make(http.Header)
	headers.Set("Authorization", "Bearer account-secret")
	headers.Set(Header, "old-state")
	headers.Set("Cookie", "old-cookie")
	started := time.Now()
	record, err := manager.acquire(ctx, headers, "model-a")
	require.NoError(test, err)
	require.Less(test, time.Since(started), 2*time.Second)
	require.Equal(test, "US", record.Country)
	require.Equal(test, "198.51.100.15", record.SourceIP)
	require.Equal(test, value, record.Value)
	_, err = manager.acquire(ctx, headers, "model-a")
	require.ErrorContains(test, err, "already used")
	trace.Store("ip=198.51.100.16\nloc=DE\n")
	_, err = manager.acquire(ctx, headers, "model-a")
	require.ErrorContains(test, err, "requested country")
	trace.Store("ip=2001:db8::1\nloc=US\n")
	_, err = manager.acquire(ctx, headers, "model-a")
	require.ErrorContains(test, err, "not IPv4")
	require.EqualValues(test, 1, calls.Load())
	require.EqualValues(test, 1, dnsCalls.Load())
}

func TestPurchasedProxyConfigurationRequiresBothRotationFields(test *testing.T) {
	valid := Config{ProxyHost: "proxy.example:7778", ProxyUsername: "prefix_{country}_{session}", ProxyPassword: "secret", ProxyUpstream: "socks5://127.0.0.1:7897"}
	require.NoError(test, validatePurchasedConfig(valid))
	invalid := valid
	invalid.ProxyUsername = "fixed-user"
	require.Error(test, validatePurchasedConfig(invalid))
	invalid = valid
	invalid.ProxyUpstream = "http://127.0.0.1:7897"
	require.Error(test, validatePurchasedConfig(invalid))
	countries, err := parseCountries("sg,US,sg,ZA")
	require.NoError(test, err)
	require.Equal(test, []string{"SG", "US", "ZA"}, countries)
}
