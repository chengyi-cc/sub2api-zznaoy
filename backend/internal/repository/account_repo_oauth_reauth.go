package repository

import (
	"context"
	"encoding/json"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"strings"
	"time"
)

// ApplyOpenAIOAuthReauth atomically swaps OAuth credentials only when the
// account still has the credential snapshot captured before protocol login.
// Re-authentication may take several minutes; the expected-value guard keeps a
// concurrent manual edit or token refresh from being overwritten by a stale
// callback. Successful swaps also restore the account's active/schedulable
// state and clear transient scheduling quarantine.
func (r *accountRepository) ApplyOpenAIOAuthReauth(
	ctx context.Context,
	taskID int64,
	workerID string,
	accountID int64,
	expectedCredentials, credentials, extra map[string]any,
) (bool, error) {
	expectedJSON, err := json.Marshal(normalizeJSONMap(expectedCredentials))
	if err != nil {
		return false, err
	}
	credentialsJSON, err := json.Marshal(normalizeJSONMap(credentials))
	if err != nil {
		return false, err
	}
	extraJSON, err := json.Marshal(normalizeJSONMap(extra))
	if err != nil {
		return false, err
	}
	var subscriptionExpiresAt *time.Time
	if raw, ok := credentials["subscription_expires_at"].(string); ok {
		if parsed, parseErr := time.Parse(time.RFC3339, strings.TrimSpace(raw)); parseErr == nil {
			subscriptionExpiresAt = &parsed
		}
	}
	result, err := r.sql.ExecContext(ctx, `
		WITH locked_task AS (
		SELECT task.id, task.account_id
		FROM openai_oauth_reauth_tasks AS task
		WHERE task.id = $9
			AND task.account_id = $3
			AND task.worker_id = $10
			AND task.status = $11
		FOR UPDATE
		), updated_account AS (
		UPDATE accounts AS a
		SET credentials = $1::jsonb,
			extra = CASE
				WHEN $2::jsonb = '{}'::jsonb THEN a.extra
				ELSE COALESCE(a.extra, '{}'::jsonb) || $2::jsonb
			END,
			expires_at = COALESCE($14::timestamptz, a.expires_at),
			status = $5,
			error_message = '',
			schedulable = TRUE,
			rate_limited_at = NULL,
			rate_limit_reset_at = NULL,
			overload_until = NULL,
			temp_unschedulable_until = NULL,
			temp_unschedulable_reason = NULL,
			updated_at = NOW()
		FROM locked_task
		WHERE a.id = $3
			AND a.id = locked_task.account_id
			AND deleted_at IS NULL
			AND platform = $6
			AND type = $7
			AND a.credentials = $4::jsonb
		RETURNING a.id
		), completed_task AS (
		UPDATE openai_oauth_reauth_tasks AS task
		SET status = $12,
			stage = $13,
			error_message = NULL,
			finished_at = NOW(),
			updated_at = NOW()
		FROM updated_account
		WHERE task.id = $9
			AND task.account_id = updated_account.id
			AND task.worker_id = $10
			AND task.status = $11
		RETURNING task.account_id
		)
		INSERT INTO scheduler_outbox (event_type, account_id, group_id, payload)
		SELECT $8, completed_task.account_id, NULL, NULL FROM completed_task
	`, string(credentialsJSON), string(extraJSON), accountID, string(expectedJSON),
		service.StatusActive, service.PlatformOpenAI, service.AccountTypeOAuth,
		service.SchedulerOutboxEventAccountChanged, taskID, workerID,
		service.OpenAIOAuthReauthStatusCallbackProcessing,
		service.OpenAIOAuthReauthStatusSucceeded, service.OpenAIOAuthReauthStageSucceeded,
		subscriptionExpiresAt)
	if err != nil {
		return false, err
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return false, err
	}
	if affected == 0 {
		return false, nil
	}
	r.syncSchedulerAccountSnapshotDetached(ctx, accountID)
	return true, nil
}
