//go:build integration

package repository

import (
	"context"
	"database/sql"
	"fmt"
	"net/url"
	"os"
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
 CREATE TABLE openai_oauth_reauth_configs (account_id BIGINT REFERENCES accounts(id), password_ciphertext TEXT);
 CREATE TABLE account_token_guard_v2_accounts (account_id BIGINT REFERENCES accounts(id));
 CREATE TABLE openai_oauth_reauth_tasks (account_id BIGINT REFERENCES accounts(id), status TEXT);
 INSERT INTO accounts VALUES (1,'original-account-token');
 INSERT INTO openai_oauth_reauth_configs VALUES (1,'unreadable-old-ciphertext');
 INSERT INTO account_token_guard_v2_accounts VALUES (1);
 `)
	require.NoError(t, err)
	repo := &openAIOAuthReauthRepository{db: db}
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
}
