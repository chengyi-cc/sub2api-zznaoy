package service

import (
	"context"
	"errors"
	"net/http"

	infraerrors "github.com/Wei-Shaw/sub2api/internal/pkg/errors"
)

var ErrCredentialEncryptionKeyMissing = errors.New("original credential encryption key missing")
var ErrCredentialRecoveryInUse = errors.New("credential recovery still has monitored accounts or active tasks")

type credentialRecoveryStore interface {
	ClearOrphanedCredentialConfigs(context.Context) error
}

// CredentialEncryptionStatus never contains a key or a ciphertext. The dedicated
// key belongs only to account re-login credentials, not user TOTP or payments.
type CredentialEncryptionStatus struct {
	Configured bool   `json:"configured"`
	Source     string `json:"source"`
}

type OpenAICredentialEncryptor interface {
	SecretEncryptor
	EncryptionStatus() (CredentialEncryptionStatus, error)
	InitializeEncryption() (CredentialEncryptionStatus, error)
}

func (s *OpenAIOAuthReauthService) CredentialEncryptionStatus() (CredentialEncryptionStatus, error) {
	if s == nil {
		return CredentialEncryptionStatus{}, infraerrors.New(http.StatusServiceUnavailable, "OPENAI_REAUTH_UNAVAILABLE", "Credential encryption is unavailable")
	}
	if manager, ok := s.encryptor.(OpenAICredentialEncryptor); ok {
		status, err := manager.EncryptionStatus()
		if err != nil {
			return CredentialEncryptionStatus{}, credentialEncryptionStorageError(err)
		}
		return status, nil
	}
	status := CredentialEncryptionStatus{Configured: s.encryptionKeyConfigured, Source: "unconfigured"}
	if status.Configured {
		status.Source = "server_config"
	}
	return status, nil
}

func (s *OpenAIOAuthReauthService) InitializeCredentialEncryption() (CredentialEncryptionStatus, error) {
	if s != nil {
		s.credentialRecoveryMu.Lock()
		defer s.credentialRecoveryMu.Unlock()
		if manager, ok := s.encryptor.(OpenAICredentialEncryptor); ok {
			status, err := manager.InitializeEncryption()
			if err != nil {
				return CredentialEncryptionStatus{}, credentialEncryptionStorageError(err)
			}
			return status, nil
		}
	}
	return CredentialEncryptionStatus{}, infraerrors.New(http.StatusServiceUnavailable, "CREDENTIAL_ENCRYPTION_UNAVAILABLE", "Credential encryption setup is unavailable")
}

// Recovery is only for a missing key after the administrator has removed all
// monitored accounts. Never rotate a readable key or touch account credentials.
func (s *OpenAIOAuthReauthService) ResetCredentialEncryption(ctx context.Context) (CredentialEncryptionStatus, error) {
	if s == nil {
		return CredentialEncryptionStatus{}, infraerrors.New(http.StatusServiceUnavailable, "CREDENTIAL_ENCRYPTION_UNAVAILABLE", "Credential recovery is unavailable")
	}
	s.credentialRecoveryMu.Lock()
	defer s.credentialRecoveryMu.Unlock()
	manager, managed := s.encryptor.(OpenAICredentialEncryptor)
	repo, recoverable := s.repo.(credentialRecoveryStore)
	if !managed || !recoverable {
		return CredentialEncryptionStatus{}, infraerrors.New(http.StatusServiceUnavailable, "CREDENTIAL_ENCRYPTION_UNAVAILABLE", "Credential recovery is unavailable")
	}
	status, err := manager.EncryptionStatus()
	if err != nil && !errors.Is(err, ErrCredentialEncryptionKeyMissing) {
		return CredentialEncryptionStatus{}, credentialEncryptionStorageError(err)
	}
	if err == nil && status.Configured {
		return CredentialEncryptionStatus{}, infraerrors.Conflict("CREDENTIAL_RECOVERY_NOT_NEEDED", "A usable encryption key already exists; it will not be replaced")
	}
	if err = repo.ClearOrphanedCredentialConfigs(ctx); err != nil {
		if errors.Is(err, ErrCredentialRecoveryInUse) {
			return CredentialEncryptionStatus{}, infraerrors.Conflict("CREDENTIAL_RECOVERY_IN_USE", "Remove all monitored accounts and finish active re-login tasks before clearing saved login credentials")
		}
		return CredentialEncryptionStatus{}, infraerrors.New(http.StatusServiceUnavailable, "CREDENTIAL_RECOVERY_CLEAR_FAILED", "Saved login credentials could not be cleared; retry after checking database availability")
	}
	status, err = manager.InitializeEncryption()
	if err != nil {
		return CredentialEncryptionStatus{}, infraerrors.New(http.StatusServiceUnavailable, "CREDENTIAL_RECOVERY_INITIALIZE_FAILED", "Old login credentials were cleared, but encryption could not be initialized. Check database availability and retry initialization")
	}
	return status, nil
}

func credentialEncryptionStorageError(cause error) error {
	if errors.Is(cause, ErrCredentialEncryptionKeyMissing) {
		return infraerrors.New(http.StatusServiceUnavailable, "CREDENTIAL_ENCRYPTION_KEY_MISSING", "Saved login credentials remain, but the original encryption key is missing. Restore the key or explicitly discard the old login credentials")
	}

	return infraerrors.New(http.StatusServiceUnavailable, "CREDENTIAL_ENCRYPTION_STORAGE_FAILED", "Cannot access the credential encryption key. Check database availability or the original encryption configuration; existing keys will not be replaced.")
}
