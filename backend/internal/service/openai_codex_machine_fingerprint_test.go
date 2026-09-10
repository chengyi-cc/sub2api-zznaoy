package service

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/model"
	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func machineTestContext(userID int64) *gin.Context {
	context, _ := gin.CreateTestContext(httptest.NewRecorder())
	context.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	context.Request.Header.Set("User-Agent", "codex_cli_rs/0.146.0")
	context.Request.Header.Set("session-id", "session-one")
	context.Request.Header.Set("thread-id", "thread-one")
	context.Request.Header.Set("x-codex-window-id", "thread-one:0")
	context.Request.Header.Set("x-codex-turn-metadata", `{"session_id":"session-one","thread_id":"thread-one","window_id":"thread-one:0","sandbox":"workspace-write"}`)
	context.Set("api_key", &APIKey{ID: userID})
	return context
}

func TestCodexMachineFingerprintStableIsolatedAndConsistent(t *testing.T) {
	account := newTestOAuthAccount(1, map[string]any{codexFingerprintModeExtraKey: "machine"})
	context := machineTestContext(10)
	ids := resolveCodexFingerprintIDsFromRequest(account, context.Request.Header)
	require.NotNil(t, ids)
	require.True(t, usesCodexMachineFingerprint(account))
	require.Equal(t, ids.machinePseudonym("session-one"), ids.machinePseudonym("session-one"))
	require.NotEqual(t, ids.machinePseudonym("session-one"), ids.machinePseudonym("session-two"))
	require.Equal(t, ids.machinePseudonym("custom-cache"), ids.machinePseudonym(" custom-cache "))
	another := machineTestContext(11)
	otherIDs := resolveCodexFingerprintIDsFromRequest(account, another.Request.Header)
	require.Equal(t, ids.machinePseudonym("session-one"), otherIDs.machinePseudonym("session-one"))
	otherAccount := newTestOAuthAccount(2, map[string]any{codexFingerprintModeExtraKey: "machine", codexFingerprintSeedExtraKey: userSuppliedCodexFingerprintSeed})
	require.NotEqual(t, ids.machinePseudonym("session-one"), resolveCodexFingerprintIDsFromRequest(otherAccount, context.Request.Header).machinePseudonym("session-one"))
	original := []byte(`{"prompt_cache_key":"session-one","input":[{"text":"untouched"}],"client_metadata":{"session_id":"session-one","thread_id":"thread-one","x-codex-window-id":"thread-one:0","x-codex-turn-metadata":"{\"session_id\":\"session-one\",\"thread_id\":\"thread-one\",\"sandbox\":\"workspace-write\"}"}}`)
	var body map[string]any
	require.NoError(t, json.Unmarshal(original, &body))
	require.True(t, applyCodexFingerprintClientMetadata(body, ids))
	raw, changed, err := applyCodexFingerprintClientMetadataRaw(original, resolveCodexFingerprintIDsFromRequest(account, context.Request.Header))
	require.NoError(t, err)
	require.True(t, changed)
	decoded, err := json.Marshal(body)
	require.NoError(t, err)
	require.JSONEq(t, string(decoded), string(raw))
	applyCodexFingerprintClientMetadata(body, ids)
	repeated, err := json.Marshal(body)
	require.NoError(t, err)
	require.JSONEq(t, string(decoded), string(repeated))
	headers := context.Request.Header.Clone()
	headers.Set("session_id", "legacy")
	headers.Set("conversation_id", "legacy")
	applyCodexFingerprintHeaders(headers, ids)
	metadata := body["client_metadata"].(map[string]any)
	require.Equal(t, body["prompt_cache_key"], metadata["session_id"])
	require.Equal(t, metadata["session_id"], headers.Get("session-id"))
	require.Equal(t, metadata["thread_id"], headers.Get("thread-id"))
	require.Equal(t, "thread-one:0", headers.Get("x-codex-window-id"))
	require.Empty(t, headers.Get("session_id"))
	require.Empty(t, headers.Get("conversation_id"))
	require.Contains(t, headers.Get("x-codex-turn-metadata"), "workspace-write")
	require.Equal(t, "session-one", context.Request.Header.Get("session-id"))
}

func TestCodexMachineNonCodexDoesNotSynthesizeSession(t *testing.T) {
	account := newTestOAuthAccount(1, map[string]any{codexFingerprintModeExtraKey: "machine"})
	context := machineTestContext(10)
	context.Request.Header = http.Header{}
	require.True(t, usesCodexMachineFingerprint(account))
	ids := resolveCodexFingerprintIDsFromRequest(account, context.Request.Header)
	body := map[string]any{"prompt_cache_key": "cache", "input": "hello"}
	require.True(t, applyCodexFingerprintClientMetadata(body, ids))
	require.NotContains(t, body, "client_metadata")
	require.NotEqual(t, "cache", body["prompt_cache_key"])
	headers := http.Header{}
	applyCodexFingerprintHeaders(headers, ids)
	require.Empty(t, headers.Get("session-id"))
	require.Empty(t, headers.Get("x-codex-window-id"))
	withoutCache := map[string]any{"input": "hello"}
	require.False(t, applyCodexFingerprintClientMetadata(withoutCache, ids))
	require.NotContains(t, withoutCache, "prompt_cache_key")
	for _, original := range []string{`[1,2,3]`, `"scalar"`, `not json`} {
		rewritten, changed, err := applyCodexFingerprintClientMetadataRaw([]byte(original), ids)
		require.NoError(t, err)
		require.False(t, changed)
		require.Equal(t, original, string(rewritten))
	}
}

func TestCodexMachineSeedLifecycleAndLegacyDefaults(t *testing.T) {
	require.Equal(t, codexFingerprintOff, (&Account{Platform: PlatformOpenAI, Type: AccountTypeOAuth}).GetCodexFingerprintMode())
	extra := prepareCodexFingerprintExtraForCreate(PlatformOpenAI, AccountTypeOAuth, map[string]any{codexFingerprintModeExtraKey: "machine"})
	seed, valid := codexFingerprintSeed(extra)
	require.True(t, valid)
	require.True(t, ShouldEnsureCodexFingerprintSeedForExtraUpdates(extra))
	account := &Account{Platform: PlatformOpenAI, Type: AccountTypeOAuth, Extra: extra}
	updated := prepareCodexFingerprintExtraForUpdate(account, map[string]any{codexFingerprintModeExtraKey: "off"})
	require.Equal(t, seed, updated[codexFingerprintSeedExtraKey])
	require.Nil(t, resolveCodexFingerprintIDs(&Account{Platform: PlatformOpenAI, Type: AccountTypeOAuth}, "session", codexFingerprintMachine))
}

func TestCodexMachineForwardPipelines(t *testing.T) {
	for _, passthrough := range []bool{false, true} {
		account := newTestOAuthAccount(1, map[string]any{codexFingerprintModeExtraKey: "machine", "openai_oauth_passthrough": passthrough})
		account.Credentials = map[string]any{"access_token": "test-token", "chatgpt_account_id": "test-account"}
		context := machineTestContext(10)
		original := []byte(`{"model":"gpt-5.4","stream":false,"instructions":"test","input":[{"role":"user","content":"hello"}],"prompt_cache_key":"session-one","client_metadata":{"session_id":"session-one","thread_id":"thread-one","x-codex-window-id":"thread-one:0"}}`)
		upstream := &httpUpstreamRecorder{err: errors.New("stop after capture")}
		gateway := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream}
		_, err := gateway.Forward(context.Request.Context(), context, account, original)
		require.Error(t, err)
		require.NotNil(t, upstream.lastReq)
		var body map[string]any
		require.NoError(t, json.Unmarshal(upstream.lastBody, &body))
		metadata := body["client_metadata"].(map[string]any)
		require.Equal(t, metadata["session_id"], body["prompt_cache_key"])
		require.Equal(t, metadata["session_id"], upstream.lastReq.Header.Get("session-id"))
		require.Equal(t, metadata["thread_id"], upstream.lastReq.Header.Get("thread-id"))
		require.Equal(t, metadata["x-codex-window-id"], upstream.lastReq.Header.Get("x-codex-window-id"))
		require.Empty(t, upstream.lastReq.Header.Get("session_id"))
		require.Empty(t, upstream.lastReq.Header.Get("conversation_id"))
	}
}

func TestCodexMachineRequestBuilders(t *testing.T) {
	for _, passthrough := range []bool{false, true} {
		account := newTestOAuthAccount(1, map[string]any{codexFingerprintModeExtraKey: "machine"})
		context := machineTestContext(10)
		ids := resolveCodexFingerprintIDsFromRequest(account, context.Request.Header)
		stageCodexFingerprintIDs(context, ids)
		body := []byte(`{"model":"gpt-5","input":"hello","prompt_cache_key":"cache"}`)
		gateway := &OpenAIGatewayService{}
		var request *http.Request
		var err error
		if passthrough {
			request, err = gateway.buildUpstreamRequestOpenAIPassthrough(context.Request.Context(), context, account, body, "token")
		} else {
			request, err = gateway.buildUpstreamRequest(context.Request.Context(), context, account, body, "token", true, "cache", true)
		}
		require.NoError(t, err)
		require.Equal(t, ids.machinePseudonym("session-one"), request.Header.Get("session-id"))
		require.Empty(t, request.Header.Get("session_id"))
		require.Empty(t, request.Header.Get("conversation_id"))
	}
}

type fingerprintTransportRecorder struct {
	HTTPUpstream
	profile       *tlsfingerprint.Profile
	ordinaryCalls int
	tlsCalls      int
}

func (recorder *fingerprintTransportRecorder) Do(request *http.Request, proxyURL string, accountID int64, concurrency int) (*http.Response, error) {
	recorder.ordinaryCalls++
	return &http.Response{StatusCode: 200}, nil
}

func (recorder *fingerprintTransportRecorder) DoWithTLS(request *http.Request, proxyURL string, accountID int64, concurrency int, profile *tlsfingerprint.Profile) (*http.Response, error) {
	recorder.tlsCalls++
	recorder.profile = profile
	return &http.Response{StatusCode: 200}, nil
}

func TestOpenAIFingerprintTLSProfileFallback(t *testing.T) {
	account := newTestOAuthAccount(1, map[string]any{"enable_tls_fingerprint": true})
	require.NotNil(t, resolveOpenAITransportTLSProfile(nil, account))
	profiles := &TLSFingerprintProfileService{}
	account.Extra["tls_fingerprint_profile_id"] = int64(99)
	require.Equal(t, "Built-in Default (Node.js 24.x)", resolveOpenAITransportTLSProfile(profiles, account).Name)
	account.Extra["enable_tls_fingerprint"] = false
	require.Nil(t, resolveOpenAITransportTLSProfile(nil, account))
	require.Nil(t, resolveOpenAITransportTLSProfile(profiles, account))
}

func TestOpenAIFingerprintTLSRouting(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		account := newTestOAuthAccount(1, map[string]any{"enable_tls_fingerprint": enabled, "tls_fingerprint_profile_id": int64(7)})
		recorder := &fingerprintTransportRecorder{}
		profile := &model.TLSFingerprintProfile{Name: "custom"}
		profiles := &TLSFingerprintProfileService{localCache: map[int64]*model.TLSFingerprintProfile{7: profile}}
		gateway := &OpenAIGatewayService{httpUpstream: recorder, tlsFPProfileService: profiles}
		request := httptest.NewRequest(http.MethodPost, "https://example.com/responses", bytes.NewBufferString("{}"))
		_, err := gateway.doOpenAIUpstream(request, "", account)
		require.NoError(t, err)
		if enabled {
			require.Equal(t, 1, recorder.tlsCalls)
			require.Equal(t, profile.ToTLSProfile(), recorder.profile)
			require.Equal(t, 0, recorder.ordinaryCalls)
		} else {
			require.Equal(t, 0, recorder.tlsCalls)
			require.Equal(t, 1, recorder.ordinaryCalls)
		}
		tester := &AccountTestService{httpUpstream: recorder, tlsFPProfileService: profiles}
		_, err = tester.doOpenAIAccountTestUpstream(request, "", account, false)
		require.NoError(t, err)
		if enabled {
			require.Equal(t, 2, recorder.tlsCalls)
		} else {
			require.Equal(t, 2, recorder.ordinaryCalls)
		}
	}
	require.False(t, (*Account)(nil).IsTLSFingerprintEnabled())
	require.False(t, (&Account{Platform: PlatformOpenAI, Type: AccountTypeAPIKey, Extra: map[string]any{"enable_tls_fingerprint": true}}).IsTLSFingerprintEnabled())
	require.True(t, (&Account{Platform: PlatformAnthropic, Type: AccountTypeSetupToken, Extra: map[string]any{"enable_tls_fingerprint": true}}).IsTLSFingerprintEnabled())
}
