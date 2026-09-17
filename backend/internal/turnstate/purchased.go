package turnstate

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"golang.org/x/net/proxy"
)

type contextDialer interface {
	DialContext(context.Context, string, string) (net.Conn, error)
}

type ProbeError struct {
	Message  string
	Status   int
	Length   int
	SourceIP string
	Country  string
}

func (failure *ProbeError) Error() string { return failure.Message }

func validatePurchasedConfig(config Config) error {
	if config.ProxyHost == "" && config.ProxyUsername == "" && config.ProxyPassword == "" {
		return nil
	}
	if _, _, err := net.SplitHostPort(config.ProxyHost); err != nil {
		return errors.New("TURN_STATE_PROXY_HOST must be host:port")
	}
	if !strings.Contains(config.ProxyUsername, "{country}") || !strings.Contains(config.ProxyUsername, "{session}") || config.ProxyPassword == "" {
		return errors.New("purchased proxy requires username country/session placeholders and a password")
	}
	if config.ProxyUpstream != "" {
		upstream, err := url.Parse(config.ProxyUpstream)
		if err != nil || (upstream.Scheme != "socks5" && upstream.Scheme != "socks5h") || upstream.Hostname() == "" || upstream.Port() == "" || upstream.Path != "" || upstream.RawQuery != "" || upstream.Fragment != "" {
			return errors.New("TURN_STATE_PROXY_UPSTREAM must be a SOCKS5 proxy URL")
		}
	}
	return nil
}

func (manager *Manager) Configured(source string) bool {
	if manager == nil {
		return false
	}
	if source == SourceIPv6 {
		return manager.config.URL != "" && manager.config.Token != ""
	}
	return manager.config.ProxyHost != "" && manager.config.ProxyUsername != "" && manager.config.ProxyPassword != ""
}

func (manager *Manager) purchasedDialer(country string) (contextDialer, error) {
	buffer := make([]byte, 4)
	if _, err := rand.Read(buffer); err != nil {
		return nil, errors.New("cannot generate proxy session")
	}
	username := strings.NewReplacer("{country}", country, "{session}", hex.EncodeToString(buffer)).Replace(manager.config.ProxyUsername)
	var forward proxy.Dialer = &net.Dialer{Timeout: 10 * time.Second}
	if manager.config.ProxyUpstream != "" {
		endpoint, _ := url.Parse(manager.config.ProxyUpstream)
		var auth *proxy.Auth
		if endpoint.User != nil {
			password, _ := endpoint.User.Password()
			auth = &proxy.Auth{User: endpoint.User.Username(), Password: password}
		}
		upstream, err := proxy.SOCKS5("tcp", endpoint.Host, auth, forward)
		if err != nil {
			return nil, errors.New("cannot configure proxy upstream")
		}
		forward = upstream
	}
	dialer, err := proxy.SOCKS5("tcp", manager.config.ProxyHost, &proxy.Auth{User: username, Password: manager.config.ProxyPassword}, forward)
	if err != nil {
		return nil, errors.New("cannot configure purchased proxy")
	}
	contextual, ok := dialer.(contextDialer)
	if !ok {
		return nil, errors.New("purchased proxy cannot enforce connection deadline")
	}
	return contextual, nil
}

func (manager *Manager) purchasedIPv4(ctx context.Context, dialer contextDialer) (string, error) {
	manager.ipMu.Lock()
	defer manager.ipMu.Unlock()
	if manager.ipv4 != "" && time.Now().Before(manager.ipv4Expires) {
		return manager.ipv4, nil
	}
	lookupCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	transport := &http.Transport{DialContext: dialer.DialContext, TLSClientConfig: manager.tlsConfig.Clone(), DisableKeepAlives: true, TLSHandshakeTimeout: 5 * time.Second, MaxResponseHeaderBytes: 16 << 10}
	defer transport.CloseIdleConnections()
	request, _ := http.NewRequestWithContext(lookupCtx, http.MethodGet, "https://cloudflare-dns.com/dns-query?name=chatgpt.com&type=A", nil)
	request.Header.Set("Accept", "application/dns-json")
	response, err := (&http.Client{Transport: transport, Timeout: 10 * time.Second, CheckRedirect: noRedirect}).Do(request)
	if err != nil {
		return "", errors.New("proxied IPv4 DNS lookup failed")
	}
	defer response.Body.Close()
	var answer struct {
		Status int
		Answer []struct {
			Type int
			TTL  int
			Data string
		}
	}
	if response.StatusCode != 200 || json.NewDecoder(io.LimitReader(response.Body, 32768)).Decode(&answer) != nil || answer.Status != 0 {
		return "", errors.New("proxied IPv4 DNS response is invalid")
	}
	for _, candidate := range answer.Answer {
		address := net.ParseIP(candidate.Data)
		if candidate.Type != 1 || address == nil || address.To4() == nil || !address.IsGlobalUnicast() || address.IsPrivate() || address.IsLoopback() {
			continue
		}
		manager.ipv4 = address.String()
		manager.ipv4Expires = time.Now().Add(time.Duration(min(max(candidate.TTL, 1), 60)) * time.Second)
		return manager.ipv4, nil
	}
	return "", errors.New("no public IPv4 target address found")
}

func (manager *Manager) acquire(ctx context.Context, headers http.Header, model string) (Record, error) {
	if sampleOptions(ctx).Source == SourceIPv6 {
		return manager.acquireIPv6(ctx, headers, model)
	}
	return manager.acquirePurchased(ctx, headers, model)
}

func (manager *Manager) acquirePurchased(parent context.Context, headers http.Header, model string) (Record, error) {
	options := sampleOptions(parent)
	if !manager.Configured(SourcePurchased) {
		return Record{}, errors.New("purchased proxy is not configured")
	}
	ctx, cancel := context.WithTimeout(parent, 30*time.Second)
	defer cancel()
	dialer, err := manager.dialPurchased(options.Country)
	if err != nil {
		return Record{}, err
	}
	address, err := manager.purchasedIPv4(ctx, dialer)
	if err != nil {
		return Record{}, err
	}
	connection, err := dialer.DialContext(ctx, "tcp", net.JoinHostPort(address, "443"))
	if err != nil {
		return Record{}, errors.New("purchased proxy connection failed")
	}
	defer connection.Close()
	stop := context.AfterFunc(ctx, func() { _ = connection.Close() })
	defer stop()
	deadline, _ := ctx.Deadline()
	_ = connection.SetDeadline(deadline)
	config := manager.tlsConfig.Clone()
	config.ServerName = "chatgpt.com"
	config.NextProtos = []string{"http/1.1"}
	secure := tls.Client(connection, config)
	defer secure.Close()
	if secure.HandshakeContext(ctx) != nil {
		return Record{}, errors.New("purchased proxy TLS handshake failed")
	}
	reader := bufio.NewReaderSize(secure, 64<<10)
	trace, _ := http.NewRequest(http.MethodGet, "https://chatgpt.com/cdn-cgi/trace", nil)
	trace.Header.Set("User-Agent", headers.Get("User-Agent"))
	if trace.Write(secure) != nil {
		return Record{}, errors.New("source verification request failed")
	}
	traceResponse, err := http.ReadResponse(reader, trace)
	if err != nil {
		return Record{}, errors.New("source verification response failed")
	}
	data, readErr := io.ReadAll(io.LimitReader(traceResponse.Body, 32769))
	if readErr != nil || len(data) > 32768 || traceResponse.StatusCode != 200 || traceResponse.Close {
		_ = secure.Close()
		_ = traceResponse.Body.Close()
		return Record{}, errors.New("source verification unavailable")
	}
	_ = traceResponse.Body.Close()
	fields := map[string]string{}
	for _, line := range strings.Split(string(data), "\n") {
		key, value, ok := strings.Cut(line, "=")
		if ok {
			fields[strings.TrimSpace(key)] = strings.TrimSpace(value)
		}
	}
	failure := &ProbeError{SourceIP: fields["ip"], Country: fields["loc"]}
	ip := net.ParseIP(failure.SourceIP)
	if ip == nil || ip.To4() == nil || failure.Country != options.Country {
		failure.Message = "proxy source is not IPv4 in the requested country"
		return Record{}, failure
	}
	digest := sha256.Sum256([]byte(ip.String()))
	claimed, err := manager.cache.SetNX(ctx, fmt.Sprintf("codex:turn-state:used-ip:%x", digest), "1", lifetime).Result()
	if err != nil {
		failure.Message = "cannot reserve verified proxy source"
		return Record{}, failure
	}
	if !claimed {
		failure.Message = "verified proxy source was already used within one hour"
		return Record{}, failure
	}
	payload, _ := json.Marshal(map[string]any{"model": model, "instructions": "Reply briefly.", "input": []any{map[string]any{"role": "user", "content": []any{map[string]any{"type": "input_text", "text": "Reply OK."}}}}, "stream": true, "store": false})
	request, _ := http.NewRequest(http.MethodPost, "https://chatgpt.com/backend-api/codex/responses", bytes.NewReader(payload))
	request.Header = sampleHeaders(headers)
	if request.Header.Get("Authorization") == "" {
		return Record{}, errors.New("account authorization unavailable")
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "text/event-stream")
	request.Header.Set("Accept-Encoding", "identity")
	session := make([]byte, 8)
	if _, err := rand.Read(session); err != nil {
		return Record{}, errors.New("cannot create sample session")
	}
	request.Header.Set("Session-Id", hex.EncodeToString(session))
	if request.Write(secure) != nil {
		failure.Message = "sample request write failed"
		return Record{}, failure
	}
	response, err := http.ReadResponse(reader, request)
	if err != nil {
		failure.Message = "sample response headers unavailable"
		return Record{}, failure
	}
	value := response.Header.Get(Header)
	failure.Status, failure.Length = response.StatusCode, len(value)
	_ = secure.Close()
	_ = response.Body.Close()
	if response.StatusCode != 200 {
		failure.Message = fmt.Sprintf("sample upstream returned status %d", response.StatusCode)
		return Record{}, failure
	}
	record, err := Parse(value, model, time.Now(), options.Profile)
	if err != nil {
		failure.Message = err.Error()
		return Record{}, failure
	}
	record.Options, record.Country, record.SourceIP = options.Options, options.Country, ip.String()
	return record, nil
}
