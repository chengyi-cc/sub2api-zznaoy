//go:build integration

package repository

import (
	"context"
	"database/sql"
	"fmt"
	"github.com/Wei-Shaw/sub2api/internal/config"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/stretchr/testify/require"
)

func TestCredentialRecoveryPostgres(t *testing.T) {
	dsn := os.Getenv("SUB2API_RECOVERY_TEST_DSN")
	if dsn == "" {
		t.Skip("isolated recovery test database is not configured")
	}
	ctx := context.Background()
	control, err := sql.Open("postgres", dsn)
	require.NoError(t, err)
	defer func() { _ = control.Close() }()
	schema := fmt.Sprintf("credential_recovery_test_%d", time.Now().UnixNano())
	_, err = control.ExecContext(ctx, "CREATE SCHEMA "+schema)
	require.NoError(t, err)
	defer func() { _, _ = control.ExecContext(ctx, "DROP SCHEMA "+schema+" CASCADE") }()
	parsed, err := url.Parse(dsn)
	require.NoError(t, err)
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	db, err := sql.Open("postgres", parsed.String())
	require.NoError(t, err)
	defer func() { _ = db.Close() }()
	_, err = db.ExecContext(ctx, `
 CREATE TABLE accounts (id BIGINT PRIMARY KEY, credentials TEXT NOT NULL);
 CREATE TABLE openai_oauth_reauth_configs (account_id BIGINT REFERENCES accounts(id), password_ciphertext TEXT, totp_secret_ciphertext TEXT, otp_url_ciphertext TEXT);
 CREATE TABLE security_secrets (key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
 CREATE TABLE account_token_guard_v2_accounts (account_id BIGINT REFERENCES accounts(id));
 CREATE TABLE openai_oauth_reauth_tasks (account_id BIGINT REFERENCES accounts(id), status TEXT);
 INSERT INTO accounts VALUES (1,'original-account-token');
 INSERT INTO openai_oauth_reauth_configs (account_id,password_ciphertext) VALUES (1,'unreadable-old-ciphertext');
 INSERT INTO account_token_guard_v2_accounts VALUES (1);
 `)
	require.NoError(t, err)
	repo := &openAIOAuthReauthRepository{db: db}
	t.Setenv("DATA_DIR", t.TempDir())
	fallback := &AESEncryptor{key: make([]byte, 32)}
	manager := NewOpenAICredentialEncryptor(&config.Config{}, fallback, db)
	_, err = manager.InitializeEncryption()
	require.ErrorIs(t, err, service.ErrCredentialEncryptionKeyMissing, "never replace a missing key while old credentials remain")

	count := func(table string) int {
		var n int
		require.NoError(t, db.QueryRowContext(ctx, "SELECT count(*) FROM "+table).Scan(&n))
		return n
	}
	require.ErrorIs(t, repo.ClearOrphanedCredentialConfigs(ctx), service.ErrCredentialRecoveryInUse)
	require.Equal(t, 1, count("openai_oauth_reauth_configs"))
	_, err = db.ExecContext(ctx, "DELETE FROM account_token_guard_v2_accounts; INSERT INTO openai_oauth_reauth_tasks VALUES (1,'queued')")
	require.NoError(t, err)
	require.ErrorIs(t, repo.ClearOrphanedCredentialConfigs(ctx), service.ErrCredentialRecoveryInUse)
	require.Equal(t, 1, count("openai_oauth_reauth_configs"))
	_, err = db.ExecContext(ctx, "UPDATE openai_oauth_reauth_tasks SET status='failed'")
	require.NoError(t, err)
	require.NoError(t, repo.ClearOrphanedCredentialConfigs(ctx))
	require.Zero(t, count("openai_oauth_reauth_configs"))
	require.Equal(t, 1, count("accounts"))
	require.Equal(t, 1, count("openai_oauth_reauth_tasks"))
	var credentials string
	require.NoError(t, db.QueryRowContext(ctx, "SELECT credentials FROM accounts WHERE id=1").Scan(&credentials))
	require.Equal(t, "original-account-token", credentials)
	require.NoError(t, repo.ClearOrphanedCredentialConfigs(ctx), "retry after cleanup is harmless")

	t.Run("database key survives container recreation without files", func(t *testing.T) {
		const workers = 8
		var wg sync.WaitGroup
		results := make(chan error, workers)
		for i := 0; i < workers; i++ {
			wg.Add(1)
			go func() { defer wg.Done(); _, initErr := manager.InitializeEncryption(); results <- initErr }()
		}
		wg.Wait()
		close(results)
		for initErr := range results {
			require.NoError(t, initErr)
		}
		require.Equal(t, 1, count("security_secrets"))
		status, statusErr := manager.EncryptionStatus()
		require.NoError(t, statusErr)
		require.Equal(t, "database", status.Source)
		_, statErr := os.Stat(filepath.Join(os.Getenv("DATA_DIR"), "secrets", "credential-operations.key"))
		require.True(t, os.IsNotExist(statErr), "no local key file or mount is created")
		ciphertext, encryptErr := manager.Encrypt("fixture-password-and-totp")
		require.NoError(t, encryptErr)
		_, saveErr := db.ExecContext(ctx, "INSERT INTO openai_oauth_reauth_configs (account_id,password_ciphertext) VALUES (1,$1)", ciphertext)
		require.NoError(t, saveErr)
		t.Setenv("DATA_DIR", t.TempDir()) // A completely different, empty container data directory.
		restarted := NewOpenAICredentialEncryptor(&config.Config{}, fallback, db)
		status, statusErr = restarted.EncryptionStatus()
		require.NoError(t, statusErr)
		require.True(t, status.Configured)
		plaintext, decryptErr := restarted.Decrypt(ciphertext)
		require.NoError(t, decryptErr)
		require.Equal(t, "fixture-password-and-totp", plaintext)
	})

	t.Run("legacy local key is preserved in the database", func(t *testing.T) {
		_, clearErr := db.ExecContext(ctx, "DELETE FROM openai_oauth_reauth_configs; DELETE FROM security_secrets")
		require.NoError(t, clearErr)
		t.Setenv("DATA_DIR", t.TempDir())
		keyPath := filepath.Join(os.Getenv("DATA_DIR"), "secrets", "credential-operations.key")
		require.NoError(t, os.MkdirAll(filepath.Dir(keyPath), 0o700))
		require.NoError(t, os.WriteFile(keyPath, []byte(strings.Repeat("ab", 32)), 0o600))
		legacy := &credentialEncryptor{keyPath: keyPath}
		ciphertext, encryptErr := legacy.Encrypt("legacy-fixture")
		require.NoError(t, encryptErr)
		_, saveErr := db.ExecContext(ctx, "INSERT INTO openai_oauth_reauth_configs (account_id,password_ciphertext) VALUES (1,$1)", ciphertext)
		require.NoError(t, saveErr)
		migrating := NewOpenAICredentialEncryptor(&config.Config{}, fallback, db)
		status, statusErr := migrating.EncryptionStatus()
		require.NoError(t, statusErr)
		require.Equal(t, "database", status.Source)
		var stored string
		require.NoError(t, db.QueryRowContext(ctx, "SELECT value FROM security_secrets WHERE key=$1", credentialDatabaseKey).Scan(&stored))
		require.Equal(t, strings.Repeat("ab", 32), stored, "import the exact original key, never rotate it")
		t.Setenv("DATA_DIR", t.TempDir())
		restarted := NewOpenAICredentialEncryptor(&config.Config{}, fallback, db)
		plaintext, decryptErr := restarted.Decrypt(ciphertext)
		require.NoError(t, decryptErr)
		require.Equal(t, "legacy-fixture", plaintext)
	})

	t.Run("invalid stored key is not overwritten", func(t *testing.T) {
		_, writeErr := db.ExecContext(ctx, "UPDATE security_secrets SET value='invalid-key' WHERE key=$1", credentialDatabaseKey)
		require.NoError(t, writeErr)
		_, initErr := manager.InitializeEncryption()
		require.Error(t, initErr)
		var stored string
		require.NoError(t, db.QueryRowContext(ctx, "SELECT value FROM security_secrets WHERE key=$1", credentialDatabaseKey).Scan(&stored))
		require.Equal(t, "invalid-key", stored)
	})

}
