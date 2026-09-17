package service

import (
	"bytes"
	"net/http"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/turnstate"
	"github.com/stretchr/testify/require"
)

func TestTurnStateAutoAccountDefaults(test *testing.T) {
	for _, accountType := range []string{AccountTypeOAuth, AccountTypeSetupToken} {
		account := &Account{Platform: PlatformOpenAI, Type: accountType}
		require.False(test, account.IsCodexTurnStateAutoEnabled())
		account.InitializeCodexTurnStateAuto()
		require.True(test, account.IsCodexTurnStateAutoEnabled())
		require.Equal(test, turnstate.ProfileTeam, account.Extra[turnstate.ProfileKey])
		require.Equal(test, turnstate.SourcePurchased, account.Extra[turnstate.SourceKey])
		account.Extra[turnstate.EnabledKey] = false
		account.Extra[turnstate.ProfileKey] = turnstate.ProfilePro
		account.Extra[turnstate.SourceKey] = turnstate.SourceIPv6
		account.InitializeCodexTurnStateAuto()
		require.False(test, account.IsCodexTurnStateAutoEnabled())
		require.Equal(test, turnstate.ProfilePro, account.Extra[turnstate.ProfileKey])
		require.Equal(test, turnstate.SourceIPv6, account.Extra[turnstate.SourceKey])
	}
	for _, account := range []*Account{nil, {Platform: PlatformOpenAI, Type: AccountTypeAPIKey}, {Platform: PlatformAnthropic, Type: AccountTypeOAuth}} {
		account.InitializeCodexTurnStateAuto()
		require.False(test, account.IsCodexTurnStateAutoEnabled())
	}
}

func TestTurnStateAutoReadsFinalMappedModelWithoutConsumingBody(test *testing.T) {
	payload := []byte("{\"input\":[{\"role\":\"user\",\"content\":\"private\"}],\"model\":\"actual-model\"}")
	request, err := http.NewRequest(http.MethodPost, "https://chatgpt.com/backend-api/codex/responses", bytes.NewReader(payload))
	require.NoError(test, err)
	require.Equal(test, "actual-model", requestTurnStateModel(request))
	require.Equal(test, int64(len(payload)), request.ContentLength)
	require.Empty(test, requestTurnStateModel(nil))
	malformed, _ := http.NewRequest(http.MethodPost, "https://example.com", bytes.NewBufferString("{\"model\":123}"))
	require.Empty(test, requestTurnStateModel(malformed))
}

func TestTurnStateAutoWebsocketCompatibilityIncludesFinalOverride(test *testing.T) {
	account := &Account{Platform: PlatformOpenAI, Type: AccountTypeOAuth, Extra: map[string]any{turnstate.EnabledKey: true}}
	first := make(http.Header)
	second := make(http.Header)
	first.Set(openAICodexTurnStateHeader, "state-a")
	second.Set(openAICodexTurnStateHeader, "state-b")
	require.NotEqual(test, normalizeOpenAIWSHandshakeCompatibility(account, first), normalizeOpenAIWSHandshakeCompatibility(account, second))
	account.Extra[turnstate.EnabledKey] = false
	require.Equal(test, normalizeOpenAIWSHandshakeCompatibility(account, first), normalizeOpenAIWSHandshakeCompatibility(account, second))
}
