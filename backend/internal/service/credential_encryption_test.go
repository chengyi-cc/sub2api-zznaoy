package service

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	infraerrors "github.com/Wei-Shaw/sub2api/internal/pkg/errors"
	"github.com/stretchr/testify/require"
)

type managedReauthTestEncryptor struct {
	reauthTestEncryptor
	ready bool
	err   error
}

func (e *managedReauthTestEncryptor) EncryptionStatus() (CredentialEncryptionStatus, error) {
	return CredentialEncryptionStatus{Configured: e.ready, Source: "local_file"}, e.err
}
func (e *managedReauthTestEncryptor) InitializeEncryption() (CredentialEncryptionStatus, error) {
	if e.err == nil {
		e.ready = true
	}
	return e.EncryptionStatus()
}

func TestCredentialEncryptionInitializationEnablesSavingWithoutRestart(t *testing.T) {
	svc, _, repo, _, _, _ := newReauthTestService("acct-1")
	svc.encryptionKeyConfigured = false
	svc.encryptor = &managedReauthTestEncryptor{}
	input := OpenAIOAuthReauthConfigInput{LoginEmail: "user@example.com", CredentialMode: OpenAIOAuthReauthModePasswordTOTP, Password: "fixture-password"}
	_, err := svc.SaveCredentialConfig(context.Background(), 42, input)
	require.Equal(t, "OPENAI_REAUTH_ENCRYPTION_KEY_REQUIRED", infraerrors.Reason(err))
	require.Nil(t, repo.config)
	status, err := svc.InitializeCredentialEncryption()
	require.NoError(t, err)
	require.True(t, status.Configured)
	_, err = svc.SaveCredentialConfig(context.Background(), 42, input)
	require.NoError(t, err)
	require.NotEmpty(t, repo.config.PasswordCiphertext)
	require.Empty(t, repo.config.TOTPSecretCiphertext) // An account without 2FA remains supported.
	raw, err := json.Marshal(status)
	require.NoError(t, err)
	require.JSONEq(t, `{"configured":true,"source":"local_file"}`, string(raw))
}

func TestCredentialEncryptionStorageFailureIsRedacted(t *testing.T) {
	svc, _, repo, _, _, _ := newReauthTestService("acct-1")
	svc.encryptor = &managedReauthTestEncryptor{err: errors.New("sensitive-file-content")}
	_, err := svc.InitializeCredentialEncryption()
	require.Equal(t, "CREDENTIAL_ENCRYPTION_STORAGE_FAILED", infraerrors.Reason(err))
	require.NotContains(t, err.Error(), "sensitive-file-content")
	_, err = svc.SaveConfig(context.Background(), 42, "user@example.com", "https://mail.example.com/code")
	require.Equal(t, "CREDENTIAL_ENCRYPTION_STORAGE_FAILED", infraerrors.Reason(err))
	require.Nil(t, repo.config)
}

type credentialResetTestRepo struct {
	OpenAIOAuthReauthRepository
	called bool
	err    error
}

func (r *credentialResetTestRepo) ClearOrphanedCredentialConfigs(context.Context) error {
	r.called = true
	return r.err
}

type credentialResetTestEncryptor struct {
	managedReauthTestEncryptor
	persistenceErr error
	initializeErr  error
	initialized    bool
}

func (e *credentialResetTestEncryptor) ValidatePersistentStorage() error { return e.persistenceErr }
func (e *credentialResetTestEncryptor) InitializeEncryption() (CredentialEncryptionStatus, error) {
	e.initialized = true
	if e.initializeErr != nil {
		return CredentialEncryptionStatus{}, e.initializeErr
	}
	e.err = nil
	e.ready = true
	return e.EncryptionStatus()
}

func TestCredentialEncryptionRecoveryGuardsAndSuccess(t *testing.T) {
	for _, tc := range []struct {
		name                                 string
		keyErr, storageErr, repoErr, initErr error
		ready, clear, init                   bool
		reason                               string
	}{
		{name: "missing key recovery", keyErr: ErrCredentialEncryptionKeyMissing, clear: true, init: true},
		{name: "retry empty setup", clear: true, init: true},
		{name: "readable key preserved", ready: true, reason: "CREDENTIAL_RECOVERY_NOT_NEEDED"},
		{name: "invalid key preserved", keyErr: errors.New("private-path-and-key"), reason: "CREDENTIAL_ENCRYPTION_STORAGE_FAILED"},
		{name: "mount required before cleanup", keyErr: ErrCredentialEncryptionKeyMissing, storageErr: ErrCredentialEncryptionNotPersistent, reason: "CREDENTIAL_ENCRYPTION_DATA_NOT_PERSISTENT"},
		{name: "active state blocks cleanup", keyErr: ErrCredentialEncryptionKeyMissing, repoErr: ErrCredentialRecoveryInUse, clear: true, reason: "CREDENTIAL_RECOVERY_IN_USE"},
		{name: "database failure blocks initialization", keyErr: ErrCredentialEncryptionKeyMissing, repoErr: errors.New("private-database-error"), clear: true, reason: "CREDENTIAL_RECOVERY_CLEAR_FAILED"},
		{name: "initialization failure remains explicit", keyErr: ErrCredentialEncryptionKeyMissing, initErr: errors.New("private-file-error"), clear: true, init: true, reason: "CREDENTIAL_RECOVERY_INITIALIZE_FAILED"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc, _, _, _, _, _ := newReauthTestService("acct-1")
			repo := &credentialResetTestRepo{err: tc.repoErr}
			encryptor := &credentialResetTestEncryptor{managedReauthTestEncryptor: managedReauthTestEncryptor{ready: tc.ready, err: tc.keyErr}, persistenceErr: tc.storageErr, initializeErr: tc.initErr}
			svc.repo = repo
			svc.encryptor = encryptor
			status, err := svc.ResetCredentialEncryption(context.Background())
			require.Equal(t, tc.clear, repo.called)
			require.Equal(t, tc.init, encryptor.initialized)
			if tc.reason == "" {
				require.NoError(t, err)
				require.True(t, status.Configured)
			} else {
				require.Equal(t, tc.reason, infraerrors.Reason(err))
				require.NotContains(t, err.Error(), "private-")
			}
		})
	}
}

func TestCredentialEncryptionMissingKeyHasSpecificReason(t *testing.T) {
	svc, _, _, _, _, _ := newReauthTestService("acct-1")
	svc.encryptor = &managedReauthTestEncryptor{err: ErrCredentialEncryptionKeyMissing}
	_, err := svc.CredentialEncryptionStatus()
	require.Equal(t, "CREDENTIAL_ENCRYPTION_KEY_MISSING", infraerrors.Reason(err))
}
