package service

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCredentialRecoveryRejectsThirdPartyEngine(t *testing.T) {
	svc, _, _, _, _, _ := newReauthTestService("acct-1")
	_, err := svc.SaveCredentialConfig(context.Background(), 42, OpenAIOAuthReauthConfigInput{
		LoginEmail: "user@example.com", CredentialMode: OpenAIOAuthReauthModePasswordTOTP,
		Password: "fixture-only", TOTPSecret: "fixture-only", Engine: OpenAIOAuthReauthEngineSessionStudio,
	})
	require.Error(t, err)
	require.Error(t, validateReauthRuntimeSettings(OpenAIOAuthReauthRuntimeSettings{Engine: OpenAIOAuthReauthEngineSessionStudio, WorkerConcurrency: 1}))
	_, _, err = svc.sessionStudioConfig(context.Background())
	require.Error(t, err)
}

func TestCredentialRecoveryCanonicalIdentityOverridesAliases(t *testing.T) {
	credentials := directReauthCredentials("wrong-alias", "wrong-user-alias", "user@example.com")
	credentials["id_token"] = reauthTestJWT(map[string]any{
		"sid": "wrong-alias", "sub": "wrong-user-alias", "email": "user@example.com",
		"https://api.openai.com/auth": map[string]any{
			"chatgpt_account_id": "acct-1", "chatgpt_user_id": "user-1", "user_id": "wrong-user-alias",
		},
	})
	info, _, err := directReauthTokenInfo(credentials, nil)
	require.NoError(t, err)
	require.Equal(t, "acct-1", info.ChatGPTAccountID)
	require.Equal(t, "user-1", info.ChatGPTUserID)
}
