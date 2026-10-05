package repository

import (
	"bytes"
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"errors"
	"os"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/service"
)

const credentialDatabaseKey = "openai_reauth_credential_key_v1"
const credentialStoredSecretsQuery = `SELECT EXISTS (SELECT 1 FROM openai_oauth_reauth_configs
 WHERE COALESCE(length(password_ciphertext), 0) > 0
 OR COALESCE(length(totp_secret_ciphertext), 0) > 0
 OR COALESCE(length(otp_url_ciphertext), 0) > 0)`

type credentialKeyReader interface {
	QueryRowContext(context.Context, string, ...any) *sql.Row
}

func readCredentialDatabaseKey(ctx context.Context, reader credentialKeyReader) (*AESEncryptor, error) {
	var value string
	err := reader.QueryRowContext(ctx, `SELECT value FROM security_secrets WHERE key = $1`, credentialDatabaseKey).Scan(&value)
	if err != nil {
		return nil, err
	}
	key, err := hex.DecodeString(strings.TrimSpace(value))
	if err != nil || len(key) != 32 {
		return nil, errors.New("invalid stored credential encryption key")
	}
	return &AESEncryptor{key: key}, nil
}

// Production uses the existing private system-secret table. The local file is
// only a migration source; its exact key is retained so old ciphertexts work.
func (e *credentialEncryptor) activeEncryptor() (*AESEncryptor, error) {
	if e.db == nil {
		return e.localEncryptor()
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	enc, err := readCredentialDatabaseKey(ctx, e.db)
	if err == nil {
		return enc, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return nil, err
	}
	if _, err = e.localEncryptor(); err != nil {
		return nil, err
	}
	if _, err = e.initializeDatabaseEncryption(); err != nil {
		return nil, err
	}
	readCtx, readCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer readCancel()
	return readCredentialDatabaseKey(readCtx, e.db)
}

func (e *credentialEncryptor) initializeDatabaseEncryption() (service.CredentialEncryptionStatus, error) {
	var empty service.CredentialEncryptionStatus
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	tx, err := e.db.BeginTx(ctx, nil)
	if err != nil {
		return empty, err
	}
	defer func() { _ = tx.Rollback() }()
	// Serialize initialization with credential writes and explicit recovery.
	if _, err = tx.ExecContext(ctx, `LOCK TABLE openai_oauth_reauth_configs IN SHARE ROW EXCLUSIVE MODE`); err != nil {
		return empty, err
	}
	_, err = readCredentialDatabaseKey(ctx, tx)
	if err == nil {
		if err = tx.Commit(); err != nil {
			return empty, err
		}
		return service.CredentialEncryptionStatus{Configured: true, Source: "database"}, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return empty, err
	}
	legacy, legacyErr := e.localEncryptor()
	if legacyErr != nil && !errors.Is(legacyErr, os.ErrNotExist) {
		return empty, legacyErr
	}
	var key []byte
	if legacy != nil {
		key = legacy.key
	} else {
		var stored bool
		if err = tx.QueryRowContext(ctx, credentialStoredSecretsQuery).Scan(&stored); err != nil {
			return empty, err
		}
		if stored {
			return empty, service.ErrCredentialEncryptionKeyMissing
		}
		key = make([]byte, 32)
		if _, err = rand.Read(key); err != nil {
			return empty, err
		}
	}
	if _, err = tx.ExecContext(ctx, `INSERT INTO security_secrets (key,value,created_at,updated_at) VALUES ($1,$2,NOW(),NOW()) ON CONFLICT (key) DO NOTHING`, credentialDatabaseKey, hex.EncodeToString(key)); err != nil {
		return empty, err
	}
	enc, err := readCredentialDatabaseKey(ctx, tx)
	if err != nil {
		return empty, err
	}
	if legacy != nil && !bytes.Equal(enc.key, legacy.key) {
		return empty, errors.New("conflicting legacy credential encryption keys")
	}
	if err = tx.Commit(); err != nil {
		return empty, err
	}
	return service.CredentialEncryptionStatus{Configured: true, Source: "database"}, nil
}
