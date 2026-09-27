package repository

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"entgo.io/ent/dialect"
	entsql "entgo.io/ent/dialect/sql"
	"github.com/DATA-DOG/go-sqlmock"
	dbent "github.com/Wei-Shaw/sub2api/ent"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/stretchr/testify/require"
)

func TestOpenAIQuotaRecoverySQLGuards(t *testing.T) {
	for _, tc := range []struct {
		name string
		rows int64
		fail bool
	}{{"observed generation", 1, false}, {"changed or excluded account", 0, false}, {"database failure", 0, true}} {
		t.Run(tc.name, func(t *testing.T) {
			matcher := sqlmock.QueryMatcherFunc(func(expected, actual string) error {
				if expected != "quota recovery" {
					return sqlmock.QueryMatcherRegexp.Match(expected, actual)
				}
				set, where, ok := strings.Cut(actual, " WHERE ")
				if !ok || !strings.HasPrefix(set, "UPDATE \"accounts\" SET ") {
					return fmt.Errorf("expected a conditional account update: %s", actual)
				}
				for _, field := range []string{"rate_limited_at", "rate_limit_reset_at"} {
					if !strings.Contains(set, "\""+field+"\" = NULL") || !strings.Contains(where, "\""+field+"\" = $") {
						return fmt.Errorf("missing observed-generation guard or clear for %s: %s", field, actual)
					}
				}
				for _, predicate := range []string{"\"id\" = $", "\"platform\" = $", "\"type\" = $", "\"deleted_at\" IS NULL", "\"parent_account_id\" IS NULL"} {
					if !strings.Contains(where, predicate) {
						return fmt.Errorf("missing account isolation predicate %s: %s", predicate, actual)
					}
				}
				for _, field := range []string{"status", "schedulable", "overload_until", "temp_unschedulable_until", "extra", "credentials"} {
					if strings.Contains(set, "\""+field+"\"") {
						return fmt.Errorf("quota recovery modified unrelated state %s", field)
					}
				}
				return nil
			})
			db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(matcher))
			require.NoError(t, err)
			client := dbent.NewClient(dbent.Driver(entsql.OpenDB(dialect.Postgres, db)))
			t.Cleanup(func() { _ = client.Close() })
			repo := newAccountRepositoryWithSQL(client, db, nil)
			limitedAt := time.Now().Add(-time.Hour).UTC().Truncate(time.Second)
			resetAt := limitedAt.Add(5 * time.Hour)
			update := mock.ExpectExec("quota recovery").WithArgs(sqlmock.AnyArg(), int64(27), service.PlatformOpenAI, service.AccountTypeOAuth, limitedAt, resetAt)
			writeErr := errors.New("database unavailable")
			if tc.fail {
				update.WillReturnError(writeErr)
			} else {
				update.WillReturnResult(sqlmock.NewResult(0, tc.rows))
				if tc.rows > 0 {
					mock.ExpectExec("INSERT INTO scheduler_outbox").WithArgs(service.SchedulerOutboxEventAccountChanged, int64(27), nil, nil, sqlmock.AnyArg()).WillReturnResult(sqlmock.NewResult(1, 1))
				}
			}
			cleared, err := repo.ClearOpenAIRateLimitIfObserved(context.Background(), 27, limitedAt, resetAt)
			if tc.fail {
				require.ErrorIs(t, err, writeErr)
			} else {
				require.NoError(t, err)
			}
			require.Equal(t, tc.rows > 0 && !tc.fail, cleared)
			require.NoError(t, mock.ExpectationsWereMet())
		})
	}
}
