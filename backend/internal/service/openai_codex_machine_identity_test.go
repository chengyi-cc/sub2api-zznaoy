package service

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestCodexMachineThirdPartyIdentityPipelines(t *testing.T) {
	for _, pipeline := range []string{"responses", "passthrough", "chat", "messages"} {
		t.Run(pipeline, func(t *testing.T) {
			account := newTestOAuthAccount(101, map[string]any{codexFingerprintModeExtraKey: "machine", "openai_oauth_passthrough": pipeline == "passthrough"})
			account.Credentials = map[string]any{"access_token": "token"}
			c := machineTestContext(10)
			c.Request.Header = http.Header{}
			upstream := &httpUpstreamRecorder{err: errors.New("capture only")}
			gateway := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream}
			responseBody := []byte(`{"model":"gpt-5.4","input":[{"role":"user","content":"hello"}]}`)
			chatBody := []byte(`{"model":"gpt-5.4","max_tokens":10,"messages":[{"role":"user","content":"hello"}]}`)
			var err error
			switch pipeline {
			case "chat":
				_, err = gateway.forwardAsChatCompletions(c.Request.Context(), c, account, chatBody, "", "", false)
			case "messages":
				_, err = gateway.ForwardAsAnthropic(c.Request.Context(), c, account, chatBody, "", "")
			default:
				_, err = gateway.Forward(c.Request.Context(), c, account, responseBody)
			}
			require.Error(t, err)
			require.NotNil(t, upstream.lastReq)
			h := upstream.lastReq.Header
			cm := gjson.GetBytes(upstream.lastBody, "client_metadata")
			thread := h.Get("thread-id")
			parsed, err := uuid.Parse(thread)
			require.NoError(t, err)
			require.Equal(t, uuid.Version(7), parsed.Version())
			require.Equal(t, thread, h.Get("session-id"))
			require.Equal(t, thread, h.Get("x-client-request-id"))
			require.Equal(t, thread, cm.Get("thread_id").String())
			require.Equal(t, thread, cm.Get("session_id").String())
			require.Equal(t, thread, gjson.GetBytes(upstream.lastBody, "prompt_cache_key").String())
			require.Equal(t, thread+":0", h.Get("x-codex-window-id"))
			require.Equal(t, h.Get("x-codex-window-id"), cm.Get("x-codex-window-id").String())
			require.NotEmpty(t, cm.Get("x-codex-installation-id").String())
			require.JSONEq(t, h.Get(openAIWSTurnMetadataHeader), cm.Get(openAIWSTurnMetadataHeader).String())
			turn := gjson.Parse(cm.Get(openAIWSTurnMetadataHeader).String())
			require.Equal(t, thread, turn.Get("thread_id").String())
			require.Equal(t, cm.Get("turn_id").String(), turn.Get("turn_id").String())
			require.Equal(t, "turn", turn.Get("request_kind").String())
			require.NotEmpty(t, turn.Get("context_window_id").String())
			require.Empty(t, h.Get("session_id"))
			require.Empty(t, h.Get("conversation_id"))
			require.Empty(t, c.Request.Header)
		})
	}
}

func TestCodexMachineSynthesizedSessionScope(t *testing.T) {
	gateway := &OpenAIGatewayService{}
	account := newTestOAuthAccount(102, map[string]any{codexFingerprintModeExtraKey: "machine"})
	body := []byte(`{"input":"hello","prompt_cache_key":"same-conversation"}`)
	resolve := func(user int64, acct *Account, raw []byte) *codexMachineIdentity {
		c := machineTestContext(user)
		c.Request.Header = http.Header{}
		ids := gateway.resolveCodexFingerprintForRequest(c, acct, raw)
		require.NotNil(t, ids)
		require.NotNil(t, ids.machineIdentity)
		return ids.machineIdentity
	}
	first := resolve(10, account, body)
	second := resolve(10, account, body)
	require.Equal(t, first.threadID, second.threadID)
	require.Equal(t, first.contextWindowID, second.contextWindowID)
	require.NotEqual(t, first.turnID, second.turnID)
	require.NotEqual(t, first.threadID, resolve(11, account, body).threadID)
	otherAccount := *account
	otherAccount.ID++ // Even accidentally duplicated seeds must not join accounts.
	require.NotEqual(t, first.threadID, resolve(10, &otherAccount, body).threadID)
	require.NotEqual(t, first.threadID, resolve(10, account, []byte(`{"input":"hello","prompt_cache_key":"another"}`)).threadID)
	require.NotEqual(t, resolve(10, account, []byte(`{"input":"hello"}`)).threadID, resolve(10, account, []byte(`{"input":"hello"}`)).threadID)
	c := machineTestContext(10)
	c.Request.Header = http.Header{}
	anonymous := []byte(`{"input":"hello"}`)
	a := gateway.resolveCodexFingerprintForRequest(c, account, anonymous)
	b := gateway.resolveCodexFingerprintForRequest(c, account, anonymous)
	require.Equal(t, a.machineIdentity.threadID, b.machineIdentity.threadID)
	c.Request.Header.Set("session-id", "shared-client-session")
	a = gateway.resolveCodexFingerprintForRequest(c, account, []byte(`{"input":"hello","client_metadata":{"thread_id":"thread-a"}}`))
	b = gateway.resolveCodexFingerprintForRequest(c, account, []byte(`{"input":"hello","client_metadata":{"thread_id":"thread-b"}}`))
	require.NotEqual(t, a.machineIdentity.threadID, b.machineIdentity.threadID)
}

func TestCodexMachineSynthesisProjectionIsIdempotent(t *testing.T) {
	gateway := &OpenAIGatewayService{}
	account := newTestOAuthAccount(103, map[string]any{codexFingerprintModeExtraKey: "machine"})
	c := machineTestContext(10)
	c.Request.Header = http.Header{}
	raw := []byte(`{"input":[{"role":"user","content":"keep"}],"client_metadata":{"unknown":9007199254740993,"x-codex-turn-metadata":"{\"unknown\":9007199254740995}"}}`)
	ids := gateway.resolveCodexFingerprintForRequest(c, account, raw)
	rewritten, changed, err := applyCodexFingerprintClientMetadataRaw(raw, ids)
	require.NoError(t, err)
	require.True(t, changed)
	twice, changed, err := applyCodexFingerprintClientMetadataRaw(rewritten, ids)
	require.NoError(t, err)
	require.False(t, changed)
	require.Equal(t, rewritten, twice)
	require.Equal(t, "9007199254740993", gjson.GetBytes(rewritten, "client_metadata.unknown").Raw)
	require.Equal(t, "9007199254740995", gjson.Get(gjson.GetBytes(rewritten, "client_metadata.x-codex-turn-metadata").String(), "unknown").Raw)
	require.Equal(t, gjson.GetBytes(raw, "input").Raw, gjson.GetBytes(rewritten, "input").Raw)
	var decoded map[string]any
	decoder := json.NewDecoder(strings.NewReader(string(raw)))
	decoder.UseNumber()
	require.NoError(t, decoder.Decode(&decoded))
	require.True(t, applyCodexFingerprintClientMetadata(decoded, ids))
	require.False(t, applyCodexFingerprintClientMetadata(decoded, ids))
	encoded, err := json.Marshal(decoded)
	require.NoError(t, err)
	require.JSONEq(t, string(rewritten), string(encoded))
	for _, raw := range []string{`[1]`, `"scalar"`, `not json`, `{"type":"session.update"}`, `{"type":"response.create","generate":false}`} {
		next, changed, err := applyCodexFingerprintClientMetadataRaw([]byte(raw), ids)
		require.NoError(t, err)
		require.False(t, changed)
		require.Equal(t, raw, string(next))
	}
}

func TestCodexMachineSynthesisGates(t *testing.T) {
	gateway := &OpenAIGatewayService{}
	for _, scenario := range []string{"official", "compact", "warmup", "session-update", "compaction", "api-key", "off", "invalid", "nil-context"} {
		t.Run(scenario, func(t *testing.T) {
			account := newTestOAuthAccount(104, map[string]any{codexFingerprintModeExtraKey: "machine"})
			c := machineTestContext(10)
			c.Request.Header = http.Header{}
			body := []byte(`{"input":"hello"}`)
			switch scenario {
			case "official":
				c.Request.Header.Set("User-Agent", "codex_cli_rs/0.153.4")
			case "compact":
				c.Request.URL.Path = "/v1/responses/compact"
			case "warmup":
				body = []byte(`{"type":"response.create","generate":false}`)
			case "session-update":
				body = []byte(`{"type":"session.update"}`)
			case "compaction":
				c.Request.Header.Set(openAIWSTurnMetadataHeader, `{"request_kind":"compaction"}`)
			case "api-key":
				account.Type = AccountTypeAPIKey
			case "off":
				account.Extra[codexFingerprintModeExtraKey] = "off"
			case "invalid":
				body = []byte(`{invalid`)
			case "nil-context":
				c = nil
				body = nil
			}
			ids := gateway.resolveCodexFingerprintForRequest(c, account, body)
			if ids != nil {
				require.Nil(t, ids.machineIdentity)
			}
		})
	}
}

func TestCodexMachineSessionStoreExpiryAndBound(t *testing.T) {
	var store codexMachineSessionStore
	now := time.Now()
	key := [32]byte{1}
	first := store.resolve(key, now)
	require.Equal(t, first.threadID, store.resolve(key, now.Add(time.Hour)).threadID)
	require.NotEqual(t, first.threadID, store.resolve(key, now.Add(time.Hour+codexMachineSessionTTL)).threadID)
	for i := 0; i <= codexMachineSessionLimit; i++ {
		next := [32]byte{byte(i), byte(i >> 8), 1}
		store.resolve(next, now.Add(2*codexMachineSessionTTL))
	}
	require.Len(t, store.entries, codexMachineSessionLimit)
	require.Equal(t, codexMachineSessionLimit, store.order.Len())
	require.NotContains(t, store.entries, key)
}

func TestCodexMachineSessionStoreConcurrent(t *testing.T) {
	var store codexMachineSessionStore
	key := [32]byte{2}
	now := time.Now()
	expected := store.resolve(key, now).threadID
	var workers sync.WaitGroup
	for i := 0; i < 16; i++ {
		workers.Go(func() { require.Equal(t, expected, store.resolve(key, now).threadID) })
	}
	workers.Wait()
	require.Len(t, store.entries, 1)
}

func TestCodexMachineWebSocketWarmupThenTurns(t *testing.T) {
	gateway := &OpenAIGatewayService{}
	c := machineTestContext(10)
	c.Request.Header = http.Header{}
	account := newTestOAuthAccount(105, map[string]any{codexFingerprintModeExtraKey: "machine"})
	gateway.stageCodexMachineFingerprintIDs(c, account, []byte(`{"type":"response.create","generate":false}`))
	require.Nil(t, stagedCodexFingerprintIDs(c, account).machineIdentity)
	body := []byte(`{"type":"response.create","input":[]}`)
	gateway.advanceCodexMachineWebSocketTurn(c, account, body)
	first := stagedCodexFingerprintIDs(c, account).machineIdentity
	require.NotNil(t, first)
	gateway.advanceCodexMachineWebSocketTurn(c, account, []byte(`{"type":"session.update"}`))
	require.Same(t, first, stagedCodexFingerprintIDs(c, account).machineIdentity)
	gateway.advanceCodexMachineWebSocketTurn(c, account, body)
	second := stagedCodexFingerprintIDs(c, account).machineIdentity
	require.Equal(t, first.threadID, second.threadID)
	require.NotEqual(t, first.turnID, second.turnID)
}
