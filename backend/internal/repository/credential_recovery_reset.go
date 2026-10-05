package repository

import (
	"context"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/service"
)

// Only the automatic re-login configuration table is cleared. Account records,
// access tokens, site TOTP, billing and completed task history are untouched.
func (r *openAIOAuthReauthRepository) ClearOrphanedCredentialConfigs(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	// Block new monitors/configurations/tasks while deciding whether cleanup is safe.
	if _, err = tx.ExecContext(ctx, `LOCK TABLE account_token_guard_v2_accounts, openai_oauth_reauth_configs, openai_oauth_reauth_tasks IN SHARE ROW EXCLUSIVE MODE`); err != nil {
		return err
	}
	var inUse bool
	if err = tx.QueryRowContext(ctx, `SELECT EXISTS (SELECT 1 FROM account_token_guard_v2_accounts) OR EXISTS (SELECT 1 FROM openai_oauth_reauth_tasks WHERE status IN ('queued','running','callback_processing'))`).Scan(&inUse); err != nil {
		return err
	}
	if inUse {
		return service.ErrCredentialRecoveryInUse
	}
	if _, err = tx.ExecContext(ctx, `DELETE FROM openai_oauth_reauth_configs`); err != nil {
		return err
	}
	return tx.Commit()
}
