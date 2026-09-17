package turnstate

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
)

func TestLiveConfiguredTurnStateManager(test *testing.T) {
	if os.Getenv("TURN_STATE_LIVE_MANAGER_TEST") != "1" {
		test.Skip("explicit opt-in required for real account sampling")
	}
	data, err := os.ReadFile(os.Getenv("TURN_STATE_LIVE_ACCOUNT_FILE"))
	if err != nil {
		test.Fatal("account file unavailable")
	}
	var exported struct {
		Accounts []struct {
			Credentials struct {
				AccessToken string `json:"access_token"`
				AccountID   string `json:"chatgpt_account_id"`
			}
		}
	}
	if json.Unmarshal(data, &exported) != nil || len(exported.Accounts) != 1 {
		test.Fatal("expected one account")
	}
	account := exported.Accounts[0].Credentials
	server := miniredis.RunT(test)
	cache := redis.NewClient(&redis.Options{Addr: server.Addr()})
	defer cache.Close()
	config := ConfigFromEnv()
	options := Options{Profile: os.Getenv("TURN_STATE_LIVE_PROFILE"), Source: os.Getenv("TURN_STATE_LIVE_SOURCE")}.Normalized()
	manager, err := New(config, cache, nil)
	if err != nil || manager == nil {
		test.Fatal("manager configuration unavailable")
	}
	defer manager.Close()
	if !manager.Configured(options.Source) {
		test.Fatal("selected source is not configured")
	}
	headers := make(http.Header)
	headers.Set("Authorization", "Bearer "+account.AccessToken)
	headers.Set("Chatgpt-Account-Id", account.AccountID)
	headers.Set("Originator", "codex-tui")
	headers.Set("Version", "0.153.4")
	headers.Set("User-Agent", "codex-tui/0.153.4 (Mac OS 26.5.0; arm64) iTerm.app/3.6.10 (codex-tui; 0.153.4)")
	ctx, cancel := context.WithTimeout(context.Background(), 320*time.Second)
	defer cancel()
	model := "gpt-6-astra"
	manager.Apply(ctx, 1, model, headers, options)
	var record Record
	for ctx.Err() == nil {
		statuses := manager.Snapshot(1)
		if len(statuses) == 1 && statuses[0].State == "unavailable" {
			break
		}
		if observed, readErr := manager.read(ctx, 1, model); readErr == nil {
			record = observed
			break
		}
		select {
		case <-ctx.Done():
		case <-time.After(200 * time.Millisecond):
		}
	}
	history, _ := manager.History(context.Background(), 1)
	for index := len(history) - 1; index >= 0; index-- {
		entry := history[index]
		test.Logf("probe profile=%s source=%s country=%s ip=%s status=%d length=%d accepted=%t error=%s", entry.Profile, entry.Source, entry.Country, entry.SourceIP, entry.Status, entry.Length, entry.Accepted, entry.Error)
	}
	if record.Value == "" {
		test.Fatal("no accepted state acquired within this run")
	}
	outbound := headers.Clone()
	if !manager.Apply(ctx, 1, model, outbound, options) || outbound.Get(Header) != record.Value {
		test.Fatal("cached state was not injected")
	}
	test.Logf("manager_verified profile=%s source=%s model=%s length=%d country=%s ip=%s expires_at=%s refresh_at=%s", options.Profile, options.Source, model, len(record.Value), record.Country, record.SourceIP, record.ExpiresAt.Format(time.RFC3339), record.ExpiresAt.Add(-refreshBefore).Format(time.RFC3339))
	if directory := os.Getenv("TURN_STATE_LIVE_SAMPLE_DIR"); directory != "" {
		identity := sha256.Sum256([]byte(account.AccountID))
		snapshot, _ := json.MarshalIndent(map[string]any{"model": model, "account_identity_hash": fmt.Sprintf("%x", identity), "state": record.Value, "verified_source_ip": record.SourceIP, "collected_at": time.Now().UTC()}, "", "  ")
		if os.MkdirAll(directory, 0700) != nil {
			test.Fatal("cannot prepare private sample directory")
		}
		name := fmt.Sprintf("production-%s-%s-%d-%s.json", options.Source, model, len(record.Value), time.Now().UTC().Format("20060102T150405.000000000Z"))
		if os.WriteFile(filepath.Join(directory, name), snapshot, 0600) != nil {
			test.Fatal("cannot save private sample")
		}
		test.Logf("private_sample_file=%s", name)
	}
}
