package repository

import (
	"context"
	"testing"
	"time"

	"entgo.io/ent/dialect"
	entsql "entgo.io/ent/dialect/sql"
	"github.com/DATA-DOG/go-sqlmock"
	dbent "github.com/Wei-Shaw/sub2api/ent"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/stretchr/testify/require"
)

func TestAstraBorrowHistoryPersistenceAndCursor(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	t.Cleanup(func() { _ = db.Close() })
	r := &settingRepository{client: dbent.NewClient(dbent.Driver(entsql.OpenDB(dialect.Postgres, db)))}
	now := time.Now()
	row := service.AstraBorrowHistory{SourceAccountID: 11, TargetAccountID: 22, Passed: true, Reason: "astra_probe_passed", CheckedAt: now}
	mock.ExpectExec("INSERT INTO astra_borrow_history").WithArgs(int64(11), int64(22), true, "astra_probe_passed", now).WillReturnResult(sqlmock.NewResult(10, 1))
	mock.ExpectExec("DELETE FROM astra_borrow_history.*LIMIT 200").WillReturnResult(sqlmock.NewResult(0, 0))
	require.NoError(t, r.AppendAstraBorrowHistory(context.Background(), row))
	mock.ExpectQuery("SELECT id,source_account_id.*id<\\$1.*ORDER BY id DESC LIMIT \\$2").WithArgs(int64(10), 50).
		WillReturnRows(sqlmock.NewRows([]string{"id", "source_account_id", "target_account_id", "passed", "reason", "checked_at"}).AddRow(9, 11, 22, true, "astra_probe_passed", now))
	rows, err := r.ListAstraBorrowHistory(context.Background(), 10, 50)
	require.NoError(t, err)
	require.Len(t, rows, 1)
	require.EqualValues(t, 9, rows[0].ID)
	require.Equal(t, "astra_probe_passed", rows[0].Reason)
	_, err = r.ListAstraBorrowHistory(context.Background(), -1, 50)
	require.Error(t, err)
	require.NoError(t, mock.ExpectationsWereMet())
}
