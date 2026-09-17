package turnstate

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

const Header = "X-Codex-Turn-State"
const EnabledKey = "codex_turn_state_auto_enabled"
const lifetime = time.Hour
const refreshBefore = 5 * time.Minute
const retryDelay = 2 * time.Minute

type Config struct {
	URL         string
	Token       string
	CAFile      string
	Attempts    int
	Concurrency int
}

func ConfigFromEnv() Config {
	config := Config{URL: strings.TrimRight(os.Getenv("TURN_STATE_POOL_URL"), "/"), Token: os.Getenv("TURN_STATE_POOL_TOKEN"), CAFile: os.Getenv("TURN_STATE_POOL_CA_FILE"), Attempts: 3, Concurrency: 4}
	if value, err := strconv.Atoi(os.Getenv("TURN_STATE_POOL_ATTEMPTS")); err == nil && value >= 1 && value <= 10 {
		config.Attempts = value
	}
	if value, err := strconv.Atoi(os.Getenv("TURN_STATE_POOL_CONCURRENCY")); err == nil && value >= 1 && value <= 16 {
		config.Concurrency = value
	}
	return config
}

type Record struct {
	Identity   string    `json:"identity"`
	Value      string    `json:"value"`
	Model      string    `json:"model"`
	IssuedAt   time.Time `json:"issued_at"`
	ExpiresAt  time.Time `json:"expires_at"`
	AcquiredAt time.Time `json:"acquired_at"`
}

type Status struct {
	Model     string     `json:"model"`
	State     string     `json:"state"`
	ExpiresAt *time.Time `json:"expires_at,omitempty"`
	LastError string     `json:"last_error,omitempty"`
}

type target struct {
	accountID int64
	model     string
	headers   http.Header
	lastSeen  time.Time
	retryAt   time.Time
	running   bool
	status    Status
}

type Prepare func(context.Context, int64, http.Header) (http.Header, bool)

type Manager struct {
	config    Config
	cache     redis.UniversalClient
	client    *http.Client
	tlsConfig *tls.Config
	prepare   Prepare
	ctx       context.Context
	cancel    context.CancelFunc
	mu        sync.Mutex
	targets   map[string]*target
	slots     chan struct{}
	workers   sync.WaitGroup
	closed    bool
	sample    func(context.Context, http.Header, string) (Record, error)
}

func New(config Config, cache redis.UniversalClient, prepare Prepare) (*Manager, error) {
	if config.URL == "" && config.Token == "" {
		return nil, nil
	}
	endpoint, err := url.Parse(config.URL)
	if err != nil || endpoint.Scheme != "https" || endpoint.Host == "" || endpoint.User != nil || endpoint.Path != "" || endpoint.RawQuery != "" || endpoint.Fragment != "" {
		return nil, errors.New("TURN_STATE_POOL_URL must be an HTTPS origin")
	}
	if len(config.Token) < 32 || cache == nil {
		return nil, errors.New("turn-state pool requires a token of at least 32 characters and Redis")
	}
	roots, err := x509.SystemCertPool()
	if err != nil {
		return nil, errors.New("cannot load system certificate roots")
	}
	if config.CAFile != "" {
		pem, readErr := os.ReadFile(config.CAFile)
		if readErr != nil || !roots.AppendCertsFromPEM(pem) {
			return nil, errors.New("cannot load turn-state pool CA certificate")
		}
	}
	if config.Attempts < 1 || config.Attempts > 10 {
		config.Attempts = 3
	}
	if config.Concurrency < 1 || config.Concurrency > 16 {
		config.Concurrency = 4
	}
	tlsConfig := &tls.Config{MinVersion: tls.VersionTLS12, RootCAs: roots}
	transport := &http.Transport{TLSClientConfig: tlsConfig, Proxy: nil, DialContext: (&net.Dialer{Timeout: 10 * time.Second}).DialContext, TLSHandshakeTimeout: 10 * time.Second, ResponseHeaderTimeout: 15 * time.Second, MaxResponseHeaderBytes: 64 << 10}
	ctx, cancel := context.WithCancel(context.Background())
	manager := &Manager{config: config, cache: cache, tlsConfig: tlsConfig, prepare: prepare, client: &http.Client{Transport: transport, Timeout: 20 * time.Second, CheckRedirect: noRedirect}, ctx: ctx, cancel: cancel, targets: make(map[string]*target), slots: make(chan struct{}, config.Concurrency)}
	manager.sample = manager.acquire
	manager.workers.Add(1)
	go manager.refreshLoop()
	return manager, nil
}

func noRedirect(_ *http.Request, _ []*http.Request) error { return http.ErrUseLastResponse }

func Parse(value, model string, now time.Time) (Record, error) {
	value = strings.TrimSpace(value)
	if len(value) != 292 {
		return Record{}, fmt.Errorf("candidate length %d is not accepted", len(value))
	}
	for _, character := range value {
		if character < 0x21 || character > 0x7e {
			return Record{}, errors.New("candidate contains invalid characters")
		}
	}
	raw, err := base64.URLEncoding.DecodeString(value)
	if err != nil {
		raw, err = base64.RawURLEncoding.DecodeString(value)
	}
	if err != nil || len(raw) < 73 || (len(raw)-57)%16 != 0 || raw[0] != 0x80 {
		return Record{}, errors.New("candidate timestamp format is invalid")
	}
	seconds := binary.BigEndian.Uint64(raw[1:9])
	if seconds < 1577836800 || seconds > 4102444800 {
		return Record{}, errors.New("candidate timestamp outside supported range")
	}
	issued := time.Unix(int64(seconds), 0).UTC()
	if issued.After(now.Add(30*time.Second)) || !issued.Add(lifetime).After(now.Add(2*time.Minute)) {
		return Record{}, errors.New("candidate is expired or has insufficient remaining lifetime")
	}
	return Record{Value: value, Model: model, IssuedAt: issued, ExpiresAt: issued.Add(lifetime), AcquiredAt: now.UTC()}, nil
}

func recordKey(accountID int64, model string) string {
	digest := sha256.Sum256([]byte(model))
	return fmt.Sprintf("codex:turn-state:v1:%d:%x", accountID, digest[:])
}

func accountIdentity(headers http.Header) string {
	account := headers.Get("Chatgpt-Account-Id")
	workspace := headers.Get("Chatgpt-Workspace-Id")
	if account == "" && workspace == "" {
		if authorization := headers.Get("Authorization"); authorization != "" {
			digest := sha256.Sum256([]byte(authorization))
			return hex.EncodeToString(digest[:])
		}
		return ""
	}
	digest := sha256.Sum256([]byte(account + "\x00" + workspace))
	return hex.EncodeToString(digest[:])
}

func (manager *Manager) read(ctx context.Context, accountID int64, model string) (Record, error) {
	data, err := manager.cache.Get(ctx, recordKey(accountID, model)).Bytes()
	if err != nil {
		return Record{}, err
	}
	var record Record
	if json.Unmarshal(data, &record) != nil || record.Model != model || len(record.Value) != 292 || !record.ExpiresAt.After(time.Now()) {
		return Record{}, errors.New("cached state is invalid or expired")
	}
	return record, nil
}

func (manager *Manager) Apply(ctx context.Context, accountID int64, model string, headers http.Header) bool {
	if manager == nil || accountID <= 0 || headers == nil {
		return false
	}
	model = strings.TrimSpace(model)
	if model == "" || len(model) > 256 {
		return false
	}
	readCtx, cancel := context.WithTimeout(ctx, 250*time.Millisecond)
	record, err := manager.read(readCtx, accountID, model)
	cancel()
	if err == nil && record.Identity != accountIdentity(headers) {
		err = errors.New("cached state is invalid or expired")
	}
	if err == nil {
		for name := range headers {
			if strings.EqualFold(name, Header) {
				delete(headers, name)
			}
		}
		headers.Set(Header, record.Value)
	}
	manager.mu.Lock()
	defer manager.mu.Unlock()
	if manager.closed {
		return err == nil
	}
	key := recordKey(accountID, model)
	entry := manager.targets[key]
	if entry == nil {
		if len(manager.targets) >= 2048 {
			for existingKey, existing := range manager.targets {
				if !existing.running && time.Since(existing.lastSeen) > 15*time.Minute {
					delete(manager.targets, existingKey)
				}
			}
			if len(manager.targets) >= 2048 {
				return err == nil
			}
		}
		entry = &target{accountID: accountID, model: model, status: Status{Model: model, State: "preparing"}}
		manager.targets[key] = entry
	}
	entry.headers = sampleHeaders(headers)
	entry.lastSeen = time.Now()
	if err == nil {
		expires := record.ExpiresAt
		entry.status.ExpiresAt = &expires
		if !entry.running {
			entry.status.State = "ready"
		}
	}
	if errors.Is(err, redis.Nil) || err == nil || (err != nil && err.Error() == "cached state is invalid or expired") {
		if err != nil || time.Until(record.ExpiresAt) <= refreshBefore {
			manager.scheduleLocked(entry)
		}
	} else {
		entry.status.State = "unavailable"
		entry.status.LastError = "state cache unavailable"
	}
	return err == nil
}

func sampleHeaders(headers http.Header) http.Header {
	result := make(http.Header)
	for _, name := range []string{"Authorization", "Chatgpt-Account-Id", "Chatgpt-Workspace-Id", "User-Agent", "Originator", "Version", "X-Codex-Beta-Features", "X-Codex-Installation-Id", "X-Codex-Window-Id"} {
		if value := headers.Get(name); value != "" {
			result.Set(name, value)
		}
	}
	return result
}

func (manager *Manager) scheduleLocked(entry *target) {
	if entry.running || time.Now().Before(entry.retryAt) || manager.closed {
		return
	}
	select {
	case manager.slots <- struct{}{}:
	default:
		return
	}
	entry.running = true
	entry.status.State = "preparing"
	if entry.status.ExpiresAt != nil && entry.status.ExpiresAt.After(time.Now()) {
		entry.status.State = "refreshing"
	}
	accountID, model, headers := entry.accountID, entry.model, entry.headers.Clone()
	manager.workers.Add(1)
	go manager.refresh(accountID, model, headers)
}

func (manager *Manager) refreshLoop() {
	defer manager.workers.Done()
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-manager.ctx.Done():
			return
		case <-ticker.C:
			manager.mu.Lock()
			for key, entry := range manager.targets {
				if time.Since(entry.lastSeen) > 15*time.Minute {
					if !entry.running {
						delete(manager.targets, key)
					}
					continue
				}
				if entry.status.ExpiresAt == nil || time.Until(*entry.status.ExpiresAt) <= refreshBefore {
					manager.scheduleLocked(entry)
				}
			}
			manager.mu.Unlock()
		}
	}
}

func (manager *Manager) refresh(accountID int64, model string, headers http.Header) {
	defer manager.workers.Done()
	defer func() { <-manager.slots }()
	ctx, cancel := context.WithTimeout(manager.ctx, 100*time.Second)
	defer cancel()
	key := recordKey(accountID, model)
	reason := ""
	var acquired *Record
	defer func() {
		manager.mu.Lock()
		defer manager.mu.Unlock()
		entry := manager.targets[key]
		if entry == nil {
			return
		}
		entry.running = false
		entry.retryAt = time.Now().Add(retryDelay)
		entry.status.LastError = reason
		if acquired != nil {
			expires := acquired.ExpiresAt
			entry.status.ExpiresAt = &expires
			entry.status.State = "ready"
		} else if entry.status.ExpiresAt != nil && entry.status.ExpiresAt.After(time.Now()) {
			entry.status.State = "ready"
		} else {
			entry.status.State = "unavailable"
		}
		if reason != "" {
			slog.Warn("turn-state acquisition deferred", "account_id", accountID, "model", model, "reason", reason)
		}
	}()
	lockID := make([]byte, 16)
	if _, err := rand.Read(lockID); err != nil {
		reason = "cannot create refresh lock"
		return
	}
	lockValue := hex.EncodeToString(lockID)
	locked, err := manager.cache.SetNX(ctx, key+":lock", lockValue, 110*time.Second).Result()
	if err != nil {
		reason = "state cache unavailable"
		return
	}
	if !locked {
		reason = "another instance is refreshing"
		return
	}
	defer func() {
		releaseCtx, releaseCancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer releaseCancel()
		_ = manager.cache.Eval(releaseCtx, "if redis.call('get',KEYS[1]) == ARGV[1] then return redis.call('del',KEYS[1]) else return 0 end", []string{key + ":lock"}, lockValue).Err()
	}()
	if cached, readErr := manager.read(ctx, accountID, model); readErr == nil && cached.Identity == accountIdentity(headers) && time.Until(cached.ExpiresAt) > refreshBefore {
		acquired = &cached
		return
	}
	if manager.prepare != nil {
		var enabled bool
		headers, enabled = manager.prepare(ctx, accountID, headers)
		if !enabled {
			reason = "account disabled or unavailable"
			return
		}
	}
	for attempt := 0; attempt < manager.config.Attempts && ctx.Err() == nil; attempt++ {
		record, acquireErr := manager.sample(ctx, headers, model)
		if acquireErr != nil {
			reason = acquireErr.Error()
			select {
			case <-ctx.Done():
				return
			case <-time.After(time.Second):
			}
			continue
		}
		if manager.prepare != nil {
			if currentHeaders, enabled := manager.prepare(ctx, accountID, headers); !enabled || accountIdentity(currentHeaders) != accountIdentity(headers) {
				reason = "account disabled during acquisition"
				return
			}
		}
		record.Identity = accountIdentity(headers)
		encoded, marshalErr := json.Marshal(record)
		if marshalErr != nil {
			reason = "cannot encode acquired state"
			return
		}
		if err := manager.cache.Set(ctx, key, encoded, time.Until(record.ExpiresAt)).Err(); err != nil {
			reason = "cannot save acquired state"
			return
		}
		acquired, reason = &record, ""
		return
	}
	if reason == "" {
		reason = "acquisition timed out"
	}
}

type lease struct {
	ID       string `json:"id"`
	ProxyURL string `json:"proxy_url"`
	IPv6     string `json:"ipv6"`
}

func (manager *Manager) poolCall(ctx context.Context, method, path string, output any) error {
	request, err := http.NewRequestWithContext(ctx, method, manager.config.URL+path, nil)
	if err != nil {
		return errors.New("cannot construct pool request")
	}
	request.Header.Set("Authorization", "Bearer "+manager.config.Token)
	response, err := manager.client.Do(request)
	if err != nil {
		return errors.New("egress pool connection failed")
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("egress pool returned status %d", response.StatusCode)
	}
	if output != nil && json.NewDecoder(io.LimitReader(response.Body, 8192)).Decode(output) != nil {
		return errors.New("egress pool response is invalid")
	}
	return nil
}

func (manager *Manager) acquire(parent context.Context, headers http.Header, model string) (Record, error) {
	ctx, cancel := context.WithTimeout(parent, 30*time.Second)
	defer cancel()
	var allocated lease
	if err := manager.poolCall(ctx, http.MethodPost, "/v1/leases", &allocated); err != nil {
		return Record{}, err
	}
	if len(allocated.ID) != 32 {
		return Record{}, errors.New("egress pool returned invalid lease")
	}
	if _, err := hex.DecodeString(allocated.ID); err != nil {
		return Record{}, errors.New("egress pool returned invalid lease")
	}
	defer func() {
		releaseCtx, releaseCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer releaseCancel()
		if err := manager.poolCall(releaseCtx, http.MethodDelete, "/v1/leases/"+allocated.ID, nil); err != nil {
			slog.Warn("turn-state lease release deferred to lease expiry")
		}
	}()
	proxyURL, err := url.Parse(allocated.ProxyURL)
	endpoint, _ := url.Parse(manager.config.URL)
	if err != nil || proxyURL.Scheme != "https" || proxyURL.Host != endpoint.Host || proxyURL.User == nil || proxyURL.User.Username() != allocated.ID || proxyURL.Path != "" || proxyURL.RawQuery != "" || proxyURL.Fragment != "" {
		return Record{}, errors.New("egress pool returned invalid proxy origin")
	}
	source := net.ParseIP(allocated.IPv6)
	if source == nil || source.To4() != nil {
		return Record{}, errors.New("egress pool returned invalid IPv6")
	}
	payload, err := json.Marshal(map[string]any{"model": model, "instructions": "Reply briefly.", "input": []any{map[string]any{"role": "user", "content": []any{map[string]any{"type": "input_text", "text": "Reply OK."}}}}, "stream": true, "store": false})
	if err != nil {
		return Record{}, errors.New("cannot construct sample request")
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://chatgpt.com/backend-api/codex/responses", bytes.NewReader(payload))
	if err != nil {
		return Record{}, errors.New("cannot construct sample request")
	}
	request.Header = sampleHeaders(headers)
	if request.Header.Get("Authorization") == "" {
		return Record{}, errors.New("account authorization unavailable")
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "text/event-stream")
	request.Header.Set("Accept-Encoding", "identity")
	session := make([]byte, 16)
	if _, err := rand.Read(session); err != nil {
		return Record{}, errors.New("cannot create sample session")
	}
	request.Header.Set("Session-Id", hex.EncodeToString(session))
	transport := &http.Transport{Proxy: http.ProxyURL(proxyURL), TLSClientConfig: manager.tlsConfig.Clone(), DisableKeepAlives: true, DialContext: (&net.Dialer{Timeout: 10 * time.Second}).DialContext, TLSHandshakeTimeout: 10 * time.Second, ResponseHeaderTimeout: 25 * time.Second, MaxResponseHeaderBytes: 64 << 10}
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: transport, Timeout: 30 * time.Second, CheckRedirect: noRedirect}
	response, err := client.Do(request)
	if err != nil {
		return Record{}, errors.New("sample request connection failed")
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return Record{}, fmt.Errorf("sample upstream returned status %d", response.StatusCode)
	}
	return Parse(response.Header.Get(Header), model, time.Now())
}

func (manager *Manager) Snapshot(accountID int64) []Status {
	result := []Status{}
	if manager == nil {
		return result
	}
	manager.mu.Lock()
	defer manager.mu.Unlock()
	for _, entry := range manager.targets {
		if entry.accountID != accountID {
			continue
		}
		status := entry.status
		if status.ExpiresAt != nil && !status.ExpiresAt.After(time.Now()) && !entry.running {
			status.State = "expired"
		}
		result = append(result, status)
	}
	return result
}

func (manager *Manager) Forget(accountID int64) {
	if manager == nil {
		return
	}
	manager.mu.Lock()
	defer manager.mu.Unlock()
	for key, entry := range manager.targets {
		if entry.accountID == accountID && !entry.running {
			delete(manager.targets, key)
		}
	}
}

func (manager *Manager) Close() {
	if manager == nil {
		return
	}
	manager.mu.Lock()
	manager.closed = true
	manager.cancel()
	manager.mu.Unlock()
	manager.workers.Wait()
	manager.client.CloseIdleConnections()
}
