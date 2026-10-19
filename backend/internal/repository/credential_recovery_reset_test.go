//go:build unit

package repository

import (
	"context"
	"errors"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/stretchr/testify/require"
)

func TestCredentialRecoveryClearsOnlyUnusedLoginConfigs(t *testing.T) {
	for _, tc := range []struct {
		name        string
		inUse, fail bool
	}{{"clear", false, false}, {"in use", true, false}, {"rollback", false, true}} {
		t.Run(tc.name, func(t *testing.T) {
			db, mock, err := sqlmock.New()
			require.NoError(t, err)
			defer func() { _ = db.Close() }()
			repo := &openAIOAuthReauthRepository{db: db}
			mock.ExpectBegin()
			mock.ExpectExec("LOCK TABLE account_token_guard_v2_accounts, openai_oauth_reauth_configs, openai_oauth_reauth_tasks IN SHARE ROW EXCLUSIVE MODE").WillReturnResult(sqlmock.NewResult(0, 0))
			mock.ExpectQuery("SELECT EXISTS .*account_token_guard_v2_accounts.*openai_oauth_reauth_tasks.*queued.*running.*callback_processing").WillReturnRows(sqlmock.NewRows([]string{"in_use"}).AddRow(tc.inUse))
			if tc.inUse {
				mock.ExpectRollback()
			} else if tc.fail {
				mock.ExpectExec("^DELETE FROM openai_oauth_reauth_configs$").WillReturnError(errors.New("database failure"))
				mock.ExpectRollback()
			} else {
				mock.ExpectExec("^DELETE FROM openai_oauth_reauth_configs$").WillReturnResult(sqlmock.NewResult(0, 2))
				mock.ExpectCommit()
			}
			err = repo.ClearOrphanedCredentialConfigs(context.Background())
			if tc.inUse {
				require.ErrorIs(t, err, service.ErrCredentialRecoveryInUse)
			} else if tc.fail {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			require.NoError(t, mock.ExpectationsWereMet())
		})
	}
}
