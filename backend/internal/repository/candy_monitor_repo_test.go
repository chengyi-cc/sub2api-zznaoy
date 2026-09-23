package repository

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/lib/pq"
	"github.com/stretchr/testify/require"
)

// Uses an explicitly supplied disposable PostgreSQL server, never the app DSN.
// Every run creates and drops its own database and applies the full migration set.
func TestCandyMonitorPostgres(t *testing.T) {
	dsn := os.Getenv("SUB2API_CANDY_TEST_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("set SUB2API_CANDY_TEST_POSTGRES_DSN to run on a disposable PostgreSQL server")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	admin, err := sql.Open("postgres", dsn)
	require.NoError(t, err)
	defer admin.Close()
	name := fmt.Sprintf("candy_test_%d", time.Now().UnixNano())
	_, err = admin.ExecContext(ctx, `CREATE DATABASE `+pq.QuoteIdentifier(name))
	require.NoError(t, err)
	defer func() {
		_, err := admin.ExecContext(context.Background(), `DROP DATABASE `+pq.QuoteIdentifier(name)+` WITH (FORCE)`)
		require.NoError(t, err)
	}()
	db, err := sql.Open("postgres", dsn+" dbname="+name)
	require.NoError(t, err)
	defer db.Close()
	require.NoError(t, ApplyMigrations(ctx, db))
	r := NewCandyMonitorRepository(db)
	settings, err := r.Settings(ctx)
	require.NoError(t, err)
	require.Equal(t, service.CandyMonitorDefaultModel, settings.ModelID)
	var first, second, group int64
	require.NoError(t, db.QueryRowContext(ctx, `INSERT INTO accounts(name,platform,type,credentials,extra) VALUES('candy one','openai','apikey','{}','{}') RETURNING id`).Scan(&first))
	require.NoError(t, db.QueryRowContext(ctx, `INSERT INTO accounts(name,platform,type,credentials,extra) VALUES('candy two','openai','apikey','{}','{}') RETURNING id`).Scan(&second))
	require.NoError(t, db.QueryRowContext(ctx, `INSERT INTO groups(name) VALUES('candy group') RETURNING id`).Scan(&group))
	_, err = db.ExecContext(ctx, `INSERT INTO account_groups(account_id,group_id) VALUES($1,$2)`, first, group)
	require.NoError(t, err)
	list, total, err := r.List(ctx, service.CandyMonitorFilter{Page: 1, PageSize: 30, GroupID: group, Search: "one"})
	require.NoError(t, err)
	require.EqualValues(t, 1, total)
	require.Len(t, list, 1)
	require.Nil(t, list[0].Latest)
	require.False(t, list[0].Enabled)
	// An ad-hoc test must not enable a scheduled plan.
	result, err := r.Begin(ctx, first, settings.ModelID, false)
	require.NoError(t, err)
	_, err = r.Begin(ctx, first, settings.ModelID, false)
	require.ErrorIs(t, err, service.ErrCandyMonitorBusy)
	list, _, err = r.List(ctx, service.CandyMonitorFilter{Page: 1, PageSize: 30, GroupID: group})
	require.NoError(t, err)
	require.False(t, list[0].Enabled)
	require.Equal(t, "running", list[0].Latest.Verdict)
	now := time.Now()
	answer := 29
	result.Verdict = "incorrect"
	result.Actual = &answer
	result.FinishedAt = &now
	result.ResponseText = "CANDY_RESULT=29"
	require.NoError(t, r.Finish(ctx, result))
	saved, err := r.Result(ctx, result.ID)
	require.NoError(t, err)
	require.Equal(t, 29, *saved.Actual)
	require.Equal(t, result.ResponseText, saved.ResponseText)
	require.NoError(t, r.Configure(ctx, []int64{first}, service.CandyMonitorConfig{Enabled: true, UseDefaults: true, ModelID: settings.ModelID, IntervalMinutes: 60}))
	custom := service.CandyMonitorConfig{Enabled: true, ModelID: "custom-text", IntervalMinutes: 17}
	require.NoError(t, r.Configure(ctx, []int64{second}, custom))
	settings.ModelID = "updated-text"
	settings.IntervalMinutes = 90
	settings.MaxResults = 10
	require.NoError(t, r.SaveSettings(ctx, settings))
	require.NoError(t, r.SetEnabled(ctx, []int64{second}, false))
	list, _, err = r.List(ctx, service.CandyMonitorFilter{Page: 1, PageSize: 30})
	require.NoError(t, err)
	byID := map[int64]service.CandyMonitorAccount{}
	for _, a := range list {
		byID[a.AccountID] = a
	}
	require.Equal(t, "updated-text", byID[first].ModelID)
	require.Equal(t, 90, byID[first].IntervalMinutes)
	require.Equal(t, "custom-text", byID[second].ModelID)
	require.Equal(t, 17, byID[second].IntervalMinutes)
	require.False(t, byID[second].Enabled)
	_, err = db.ExecContext(ctx, `UPDATE candy_monitor_accounts SET next_run_at=NOW()-INTERVAL '1 minute'`)
	require.NoError(t, err)
	due, err := r.Due(ctx, 4)
	require.NoError(t, err)
	require.Len(t, due, 1)
	require.Equal(t, first, due[0].AccountID)
	settings.Enabled = false
	require.NoError(t, r.SaveSettings(ctx, settings))
	_, err = r.Begin(ctx, first, "updated-text", true)
	require.ErrorIs(t, err, service.ErrCandyMonitorBusy)
	due, err = r.Due(ctx, 4)
	require.NoError(t, err)
	require.Empty(t, due)
	settings.Enabled = true
	require.NoError(t, r.SaveSettings(ctx, settings))
	scheduled, err := r.Begin(ctx, first, "stale-model-from-due-scan", true)
	require.NoError(t, err)
	require.Equal(t, "updated-text", scheduled.ModelID)
	require.Equal(t, "scheduled", scheduled.Source)
	scheduled.Verdict = "pass"
	scheduled.FinishedAt = &now
	require.NoError(t, r.Finish(ctx, scheduled))
	due, err = r.Due(ctx, 4)
	require.NoError(t, err)
	require.Empty(t, due, "a scheduled claim must advance the next due time")
	// A dead process is marked inconclusive, and cannot overwrite a newer worker.
	old, err := r.Begin(ctx, first, "updated-text", false)
	require.NoError(t, err)
	_, err = db.ExecContext(ctx, `UPDATE candy_monitor_accounts SET lease_until=NOW()-INTERVAL '1 second' WHERE account_id=$1`, first)
	require.NoError(t, err)
	_, err = db.ExecContext(ctx, `UPDATE candy_monitor_results SET started_at=NOW()-INTERVAL '5 minutes' WHERE id=$1`, old.ID)
	require.NoError(t, err)
	_, err = r.Due(ctx, 4)
	require.NoError(t, err)
	expired, err := r.Result(ctx, old.ID)
	require.NoError(t, err)
	require.Equal(t, "inconclusive", expired.Verdict)
	current, err := r.Begin(ctx, first, "updated-text", false)
	require.NoError(t, err)
	old.Verdict = "pass"
	require.NoError(t, r.Finish(ctx, old))
	stillRunning, err := r.Result(ctx, current.ID)
	require.NoError(t, err)
	require.Equal(t, "running", stillRunning.Verdict)
	current.Verdict = "pass"
	current.FinishedAt = &now
	require.NoError(t, r.Finish(ctx, current))
	for i := 0; i < 12; i++ {
		v, e := r.Begin(ctx, first, "updated-text", false)
		require.NoError(t, e)
		v.Verdict = "pass"
		v.FinishedAt = &now
		require.NoError(t, r.Finish(ctx, v))
	}
	history, err := r.History(ctx, first, 50)
	require.NoError(t, err)
	require.Len(t, history, 10)
	require.Greater(t, history[0].ID, history[9].ID)
	// Account deletion cascades only its monitor data.
	_, err = db.ExecContext(ctx, `DELETE FROM accounts WHERE id=$1`, first)
	require.NoError(t, err)
	history, err = r.History(ctx, first, 50)
	require.NoError(t, err)
	require.Empty(t, history)
}
