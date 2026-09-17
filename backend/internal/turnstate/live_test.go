package turnstate

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
)

func TestLivePoolConcurrentLeases(test *testing.T) {
	if os.Getenv("TURN_STATE_LIVE_POOL_TEST") != "1" {
		test.Skip("explicit opt-in required; allocates real IPv6 leases")
	}
	server := miniredis.RunT(test)
	cache := redis.NewClient(&redis.Options{Addr: server.Addr()})
	defer cache.Close()
	config := ConfigFromEnv()
	if config.URL == "" {
		test.Fatal("pool not configured")
	}
	manager, err := New(config, cache, nil)
	if err != nil {
		test.Fatal("manager initialization failed")
	}
	defer manager.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	addresses := make(map[string]bool)
	leaseIDs := make(map[string]bool)
	for round := 1; round <= 2; round++ {
		allocated := make([]lease, 10)
		failures := make([]error, 10)
		var workers sync.WaitGroup
		start := make(chan struct{})
		started := time.Now()
		for index := range allocated {
			workers.Add(1)
			go func(index int) {
				defer workers.Done()
				<-start
				failures[index] = manager.poolCall(ctx, http.MethodPost, "/v1/leases", &allocated[index])
			}(index)
		}
		close(start)
		workers.Wait()
		accepted := 0
		for index, entry := range allocated {
			if failures[index] != nil {
				test.Errorf("allocation round=%d index=%d error=%s", round, index, failures[index])
				continue
			}
			accepted++
			if addresses[entry.IPv6] || leaseIDs[entry.ID] {
				test.Error("duplicate address or lease")
			}
			addresses[entry.IPv6] = true
			leaseIDs[entry.ID] = true
		}
		test.Logf("lease_burst round=%d success=%d concurrency=10 duration_ms=%d", round, accepted, time.Since(started).Milliseconds())
		for _, entry := range allocated {
			if entry.ID == "" {
				continue
			}
			workers.Add(1)
			go func(entry lease) {
				defer workers.Done()
				if liveRelease(manager, entry) != nil {
					test.Error("lease retirement failed")
				}
			}(entry)
		}
		workers.Wait()
		time.Sleep(2 * time.Second)
	}
	var status map[string]any
	if manager.poolCall(ctx, http.MethodGet, "/v1/status", &status) != nil {
		test.Fatal("pool status unavailable")
	}
	encoded, _ := json.Marshal(status)
	test.Logf("pool_concurrency_summary unique_ipv6=%d unique_leases=%d pool_after=%s", len(addresses), len(leaseIDs), encoded)
	if len(addresses) != 20 || len(leaseIDs) != 20 {
		test.Error("expected 20 unique leases")
	}
}

type liveResult struct {
	Status       int
	HeaderLength int
	Completed    bool
	Output       string
	ErrorCode    string
	Milliseconds int64
	Injected     bool
	state        string
}

func liveSafeCode(value string) string {
	if len(value) > 80 {
		return "redacted"
	}
	for _, character := range value {
		if !(character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z' || character >= '0' && character <= '9' || character == '_' || character == '.' || character == '-') {
			return "redacted"
		}
	}
	return value
}

func liveRelease(manager *Manager, allocated lease) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return manager.poolCall(ctx, http.MethodDelete, "/v1/leases/"+allocated.ID, nil)
}

func liveRequest(ctx context.Context, manager *Manager, allocated lease, headers http.Header, model, prompt, expectedState string) (result liveResult) {
	started := time.Now()
	defer func() { result.Milliseconds = time.Since(started).Milliseconds() }()
	endpoint, err := url.Parse(allocated.ProxyURL)
	origin, _ := url.Parse(manager.config.URL)
	if err != nil || endpoint.Scheme != "https" || endpoint.Host != origin.Host || endpoint.User == nil || endpoint.User.Username() != allocated.ID {
		result.ErrorCode = "invalid_proxy_origin"
		return
	}
	transport := &http.Transport{Proxy: http.ProxyURL(endpoint), TLSClientConfig: manager.tlsConfig.Clone(), DisableKeepAlives: true, DialContext: (&net.Dialer{Timeout: 10 * time.Second}).DialContext, TLSHandshakeTimeout: 10 * time.Second, ResponseHeaderTimeout: 30 * time.Second}
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: transport, Timeout: 65 * time.Second, CheckRedirect: noRedirect}
	payload, _ := json.Marshal(map[string]any{"model": model, "instructions": "Reply briefly.", "input": []any{map[string]any{"role": "user", "content": []any{map[string]any{"type": "input_text", "text": prompt}}}}, "stream": true, "store": false})
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://chatgpt.com/backend-api/codex/responses", bytes.NewReader(payload))
	if err != nil {
		result.ErrorCode = "request_construction_failed"
		return
	}
	request.Header = headers.Clone()
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "text/event-stream")
	request.Header.Set("Accept-Encoding", "identity")
	session := make([]byte, 16)
	if _, err = rand.Read(session); err != nil {
		result.ErrorCode = "random_failed"
		return
	}
	request.Header.Set("Session-Id", hex.EncodeToString(session))
	result.Injected = expectedState != "" && request.Header.Get(Header) == expectedState
	response, err := client.Do(request)
	if err != nil {
		result.ErrorCode = "connection_failed"
		return
	}
	defer response.Body.Close()
	result.Status = response.StatusCode
	result.state = response.Header.Get(Header)
	result.HeaderLength = len(result.state)
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		var failure struct {
			Error struct {
				Code    string
				Type    string
				Message string
			}
			Detail  string
			Message string
		}
		_ = json.NewDecoder(io.LimitReader(response.Body, 65536)).Decode(&failure)
		result.ErrorCode = liveSafeCode(failure.Error.Code)
		if result.ErrorCode == "" {
			result.ErrorCode = liveSafeCode(failure.Error.Type)
		}
		if result.ErrorCode == "" {
			message := strings.ToLower(failure.Error.Message + " " + failure.Detail + " " + failure.Message)
			if strings.Contains(message, "model") && (strings.Contains(message, "not supported") || strings.Contains(message, "does not exist") || strings.Contains(message, "not available")) {
				result.ErrorCode = "model_not_supported_or_available"
			} else {
				result.ErrorCode = "upstream_request_rejected"
			}
		}
		return
	}
	scanner := bufio.NewScanner(io.LimitReader(response.Body, 4<<20))
	scanner.Buffer(make([]byte, 4096), 1<<20)
	var output strings.Builder
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		var event struct {
			Type     string
			Delta    string
			Error    struct{ Code string }
			Code     string
			Response struct {
				Status string
				Error  struct{ Code string }
			}
		}
		if json.Unmarshal([]byte(strings.TrimSpace(strings.TrimPrefix(line, "data:"))), &event) != nil {
			continue
		}
		switch event.Type {
		case "response.output_text.delta":
			if output.Len() < 4096 {
				output.WriteString(event.Delta)
			}
		case "response.completed":
			result.Completed = event.Response.Status == "completed"
		case "response.failed", "response.incomplete", "error":
			result.ErrorCode = liveSafeCode(event.Response.Error.Code)
			if result.ErrorCode == "" {
				result.ErrorCode = liveSafeCode(event.Error.Code)
			}
			if result.ErrorCode == "" {
				result.ErrorCode = liveSafeCode(event.Code)
			}
			if result.ErrorCode == "" {
				result.ErrorCode = event.Type
			}
		}
	}
	if scanner.Err() != nil {
		result.ErrorCode = "stream_read_failed"
	}
	result.Output = output.String()
	return
}

func TestLiveAccountTurnState(test *testing.T) {
	if os.Getenv("TURN_STATE_LIVE_TEST") != "1" {
		test.Skip("explicit opt-in required; sends real account requests")
	}
	data, err := os.ReadFile(os.Getenv("TURN_STATE_LIVE_ACCOUNT_FILE"))
	if err != nil {
		test.Fatal("account file unavailable")
	}
	var exported struct {
		Accounts []struct {
			Platform    string
			Type        string
			Credentials struct {
				AccessToken string            `json:"access_token"`
				AccountID   string            `json:"chatgpt_account_id"`
				Mapping     map[string]string `json:"model_mapping"`
			}
		}
	}
	if json.Unmarshal(data, &exported) != nil || len(exported.Accounts) != 1 {
		test.Fatal("expected exactly one exported account")
	}
	account := exported.Accounts[0]
	if account.Platform != "openai" || account.Credentials.AccessToken == "" || account.Credentials.AccountID == "" {
		test.Fatal("unsupported account or missing credentials")
	}
	model := os.Getenv("TURN_STATE_LIVE_MODEL")
	if model == "" {
		model = "gpt-5.5"
	}
	if actual, exists := account.Credentials.Mapping[model]; exists {
		model = actual
	}
	server := miniredis.RunT(test)
	cache := redis.NewClient(&redis.Options{Addr: server.Addr()})
	defer cache.Close()
	config := ConfigFromEnv()
	if config.URL == "" {
		test.Fatal("pool not configured")
	}
	manager, err := New(config, cache, nil)
	if err != nil {
		test.Fatal("manager initialization failed")
	}
	defer manager.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 6*time.Minute)
	defer cancel()
	var initial map[string]any
	if err = manager.poolCall(ctx, http.MethodGet, "/v1/status", &initial); err != nil {
		test.Fatal(err)
	}
	status, _ := json.Marshal(initial)
	test.Logf("pool_before=%s model=%s", status, model)
	defer func() {
		finalCtx, finalCancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer finalCancel()
		var final map[string]any
		if manager.poolCall(finalCtx, http.MethodGet, "/v1/status", &final) != nil {
			test.Error("final pool status unavailable")
			return
		}
		encoded, _ := json.Marshal(final)
		test.Logf("pool_after=%s", encoded)
	}()
	headers := make(http.Header)
	headers.Set("Authorization", "Bearer "+account.Credentials.AccessToken)
	headers.Set("Chatgpt-Account-Id", account.Credentials.AccountID)
	headers.Set("Originator", "codex-tui")
	headers.Set("User-Agent", "codex-tui/0.153.4 (Mac OS 26.5.0; arm64) iTerm.app/3.6.10 (codex-tui; 0.153.4)")
	headers.Set("Version", "0.153.4")
	var probeLease lease
	var probe liveResult
	for attempt := 1; attempt <= 3; attempt++ {
		if err = manager.poolCall(ctx, http.MethodPost, "/v1/leases", &probeLease); err != nil {
			test.Fatal(err)
		}
		probe = liveRequest(ctx, manager, probeLease, headers, model, "Reply with exactly OK.", "")
		if err = liveRelease(manager, probeLease); err != nil {
			test.Error("probe lease release failed")
		}
		test.Logf("authentication attempt=%d status=%d completed=%t header_length=%d error_code=%s duration_ms=%d output_chars=%d", attempt, probe.Status, probe.Completed, probe.HeaderLength, probe.ErrorCode, probe.Milliseconds, len(probe.Output))
		if (probe.Status >= 400 && probe.Status < 500) || probe.Completed {
			break
		}
		time.Sleep(time.Second)
	}
	if probe.Status == http.StatusUnauthorized {
		test.Fatal("AUTHENTICATION_401: stopping before acquisition and concurrency")
	}
	if probe.Status != http.StatusOK || (probe.ErrorCode != "" && probe.ErrorCode != "server_is_overloaded") {
		test.Fatal("authentication probe failed; stopping")
	}
	var record Record
	poolOptions := Options{Profile: ProfileTeam, Source: SourceIPv6}
	poolContext := context.WithValue(ctx, acquisitionKey{}, acquisitionOptions{Options: poolOptions})
	for attempt := 1; attempt <= config.Attempts; attempt++ {
		started := time.Now()
		record, err = manager.acquire(poolContext, headers, model)
		if err == nil {
			test.Logf("production_acquire attempt=%d accepted_length=%d duration_ms=%d", attempt, len(record.Value), time.Since(started).Milliseconds())
			break
		}
		test.Logf("production_acquire attempt=%d error=%s duration_ms=%d", attempt, err, time.Since(started).Milliseconds())
		if strings.Contains(err.Error(), "status 401") || strings.Contains(err.Error(), "status 429") {
			break
		}
	}
	if err != nil {
		test.Fatal("no accepted state acquired; concurrency not started")
	}
	record.Identity = accountIdentity(headers)
	encoded, _ := json.Marshal(record)
	if cache.Set(ctx, recordKey(1, model), encoded, time.Until(record.ExpiresAt)).Err() != nil {
		test.Fatal("isolated cache write failed")
	}
	leases := make([]lease, 0, 10)
	defer func() {
		for _, allocated := range leases {
			if liveRelease(manager, allocated) != nil {
				test.Error("concurrency lease release failed")
			}
		}
	}()
	unique := map[string]bool{probeLease.IPv6: true}
	for index := 0; index < 10; index++ {
		var allocated lease
		if err = manager.poolCall(ctx, http.MethodPost, "/v1/leases", &allocated); err != nil {
			test.Fatal(err)
		}
		leases = append(leases, allocated)
		if unique[allocated.IPv6] {
			test.Fatal("pool reused IPv6")
		}
		unique[allocated.IPv6] = true
	}
	results := make([]liveResult, 10)
	expected := make([]string, 10)
	start := make(chan struct{})
	var workers sync.WaitGroup
	for index := range results {
		random := make([]byte, 6)
		if _, err = rand.Read(random); err != nil {
			test.Fatal("random failed")
		}
		expected[index] = "OK-" + hex.EncodeToString(random)
		workers.Add(1)
		go func(index int) {
			defer workers.Done()
			<-start
			outbound := headers.Clone()
			outbound.Set(Header, "old-test-state")
			if !manager.Apply(ctx, 1, model, outbound, poolOptions) {
				results[index].ErrorCode = "cache_apply_failed"
				return
			}
			results[index] = liveRequest(ctx, manager, leases[index], outbound, model, "Reply with exactly "+expected[index]+" and nothing else.", record.Value)
		}(index)
	}
	started := time.Now()
	close(start)
	workers.Wait()
	completed := 0
	for index, result := range results {
		matched := strings.TrimSpace(result.Output) == expected[index]
		test.Logf("concurrent_request=%02d status=%d completed=%t injected=%t reply_match=%t response_header_length=%d error_code=%s duration_ms=%d", index+1, result.Status, result.Completed, result.Injected, matched, result.HeaderLength, result.ErrorCode, result.Milliseconds)
		if result.Status == 200 && result.Completed && result.Injected && matched && result.ErrorCode == "" {
			completed++
		}
	}
	test.Logf("concurrency_summary success=%d total=10 unique_business_ipv6=%d duration_ms=%d", completed, len(leases), time.Since(started).Milliseconds())
	if completed != 10 {
		test.Error("one or more concurrent requests failed")
	}
}
