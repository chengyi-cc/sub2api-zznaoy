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
	require.Empty(t, list[0].History)
	require.NotNil(t, list[0].History)
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
	require.Empty(t, list[0].History, "history dots only represent completed tests")
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
	answer21 := 21
	scheduled.Actual = &answer21
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
	current.Actual = &answer21
	current.FinishedAt = &now
	require.NoError(t, r.Finish(ctx, current))
	for i := 0; i < 12; i++ {
		v, e := r.Begin(ctx, first, "updated-text", false)
		require.NoError(t, e)
		v.Verdict = "pass"
		v.Actual = &answer21
		v.FinishedAt = &now
		require.NoError(t, r.Finish(ctx, v))
	}
	history, err := r.History(ctx, first, 50)
	require.NoError(t, err)
	require.Len(t, history, 10)
	require.Greater(t, history[0].ID, history[9].ID)
	// Lifetime counters survive retention and idempotent/stale Finish calls.
	require.NoError(t, r.Finish(ctx, current))
	_, err = r.Due(ctx, 4)
	require.NoError(t, err)
	list, _, err = r.List(ctx, service.CandyMonitorFilter{Page: 1, PageSize: 30, GroupID: group})
	require.NoError(t, err)
	require.EqualValues(t, 16, list[0].TotalTests)
	require.EqualValues(t, 14, list[0].Answer21Count)
	require.EqualValues(t, 1, list[0].Answer29Count)
	require.EqualValues(t, 1, list[0].InconclusiveCount)
	large, err := r.Begin(ctx, first, "updated-text", false)
	require.NoError(t, err)
	largeAnswer := 9999999999
	large.Actual, large.Verdict, large.FinishedAt = &largeAnswer, "incorrect", &now
	require.NoError(t, r.Finish(ctx, large))
	list, _, err = r.List(ctx, service.CandyMonitorFilter{Page: 1, PageSize: 30, GroupID: group})
	require.NoError(t, err)
	require.EqualValues(t, 17, list[0].TotalTests)
	require.EqualValues(t, 1, list[0].OtherAnswerCount)
	require.Len(t, list[0].History, 10)
	require.Equal(t, large.ID, list[0].History[0].ID)
	require.Equal(t, largeAnswer, *list[0].History[0].Actual)
	for i, item := range list[0].History {
		require.Empty(t, item.ResponseText, "list results must not include large model responses")
		if i > 0 {
			require.Greater(t, list[0].History[i-1].ID, item.ID)
		}
	}
	// Filters apply to the database result and total, not just the current page.
	var third int64
	require.NoError(t, db.QueryRowContext(ctx, `INSERT INTO accounts(name,platform,type,status,credentials,extra) VALUES('candy three','anthropic','oauth','inactive','{}','{}') RETURNING id`).Scan(&third))
	_, err = db.ExecContext(ctx, `UPDATE accounts SET extra='{"privacy_mode":"training_off"}' WHERE id=$1`, first)
	require.NoError(t, err)
	enabled, paused := true, false
	for _, tc := range []struct {
		name   string
		filter service.CandyMonitorFilter
		ids    []int64
	}{
		{"combined", service.CandyMonitorFilter{GroupID: group, Platform: "openai", Type: "apikey", Status: "active", Enabled: &enabled, Verdict: "incorrect", PrivacyMode: "training_off", Search: "one"}, []int64{first}},
		{"paused includes never configured", service.CandyMonitorFilter{Enabled: &paused}, []int64{third, second}},
		{"enabled", service.CandyMonitorFilter{Enabled: &enabled}, []int64{first}},
		{"legacy enabled", service.CandyMonitorFilter{EnabledOnly: true}, []int64{first}},
		{"ungrouped", service.CandyMonitorFilter{Ungrouped: true}, []int64{third, second}},
		{"platform and type", service.CandyMonitorFilter{Platform: "anthropic", Type: "oauth"}, []int64{third}},
		{"inactive", service.CandyMonitorFilter{Status: "inactive"}, []int64{third}},
		{"privacy unset", service.CandyMonitorFilter{PrivacyMode: "__unset__"}, []int64{third, second}},
		{"untested", service.CandyMonitorFilter{Verdict: "untested"}, []int64{third, second}},
		{"no match", service.CandyMonitorFilter{Platform: "gemini"}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tc.filter.Page, tc.filter.PageSize = 1, 100
			items, total, err := r.List(ctx, tc.filter)
			require.NoError(t, err)
			require.EqualValues(t, len(tc.ids), total)
			require.Len(t, items, len(tc.ids))
			for i, id := range tc.ids {
				require.Equal(t, id, items[i].AccountID)
			}
		})
	}
	items, total, err := r.List(ctx, service.CandyMonitorFilter{Page: 2, PageSize: 1, Enabled: &paused})
	require.NoError(t, err)
	require.EqualValues(t, 2, total)
	require.Len(t, items, 1)
	require.Equal(t, second, items[0].AccountID)
	// Availability uses the same precedence as the official account list.
	for _, tc := range []struct{ update, status string }{
		{`schedulable=FALSE`, "unschedulable"},
		{`rate_limit_reset_at=NOW()+INTERVAL '1 hour'`, "rate_limited"},
		{`temp_unschedulable_until=NOW()+INTERVAL '1 hour'`, "temp_unschedulable"},
	} {
		_, err = db.ExecContext(ctx, `UPDATE accounts SET `+tc.update+` WHERE id=$1`, first)
		require.NoError(t, err)
		items, total, err = r.List(ctx, service.CandyMonitorFilter{Page: 1, PageSize: 30, Status: tc.status})
		require.NoError(t, err)
		require.EqualValues(t, 1, total)
		require.Equal(t, first, items[0].AccountID)
		items, _, err = r.List(ctx, service.CandyMonitorFilter{Page: 1, PageSize: 30, Status: "active", GroupID: group})
		require.NoError(t, err)
		require.Empty(t, items)
	}
	// Account deletion cascades only its monitor data.
	states, err := r.AccountStates(ctx, []int64{first, second, third})
	require.NoError(t, err)
	require.Len(t, states, 3)
	require.Equal(t, largeAnswer, *states[0].LastValidAnswer)
	require.Len(t, states[0].History, 10)
	require.Equal(t, large.ID, states[0].History[0].ID)
	for i, item := range states[0].History {
		require.Empty(t, item.ResponseText)
		require.NotEqual(t, "running", item.Verdict)
		if i > 0 {
			require.Greater(t, states[0].History[i-1].ID, item.ID)
		}
	}
	require.NotNil(t, states[2].History)
	require.Empty(t, states[2].History)
	require.Nil(t, states[1].LastValidAnswer)
	// New opt-in is immediately due, re-enabling preserves custom model/interval.
	require.NoError(t, r.SetMonitoring(ctx, third, true))
	require.NoError(t, r.SetMonitoring(ctx, second, true))
	states, err = r.AccountStates(ctx, []int64{second, third})
	require.NoError(t, err)
	require.True(t, states[0].Enabled)
	require.False(t, states[0].UseDefaults)
	require.Equal(t, "custom-text", states[0].ModelID)
	require.Equal(t, 17, states[0].IntervalMinutes)
	require.True(t, states[1].UseDefaults)
	require.Equal(t, settings.ModelID, states[1].ModelID)
	var dueNow bool
	require.NoError(t, db.QueryRowContext(ctx, `SELECT next_run_at<=NOW() FROM candy_monitor_accounts WHERE account_id=$1`, third).Scan(&dueNow))
	require.True(t, dueNow)
	_, err = db.ExecContext(ctx, `UPDATE candy_monitor_accounts SET next_run_at=NOW()+INTERVAL '1 hour' WHERE account_id=$1`, third)
	require.NoError(t, err)
	require.NoError(t, r.SetMonitoring(ctx, third, true))
	require.NoError(t, db.QueryRowContext(ctx, `SELECT next_run_at<=NOW() FROM candy_monitor_accounts WHERE account_id=$1`, third).Scan(&dueNow))
	require.False(t, dueNow, "retrying enabled=true must not trigger extra tests")
	require.NoError(t, r.SetMonitoring(ctx, second, false))
	// A valid 29 remains red through failures, lease recovery, and retention.
	last29, err := r.Begin(ctx, first, settings.ModelID, false)
	require.NoError(t, err)
	last29.Actual, last29.Verdict, last29.FinishedAt = &answer, "incorrect", &now
	require.NoError(t, r.Finish(ctx, last29))
	for i := 0; i < 12; i++ {
		v, err := r.Begin(ctx, first, settings.ModelID, false)
		require.NoError(t, err)
		v.Verdict, v.FinishedAt = "inconclusive", &now
		if i%2 == 0 {
			v.Verdict = "invalid_format"
		}
		require.NoError(t, r.Finish(ctx, v))
	}
	states, err = r.AccountStates(ctx, []int64{first})
	require.NoError(t, err)
	require.Equal(t, 29, *states[0].LastValidAnswer)
	require.WithinDuration(t, last29.StartedAt, *states[0].LastValidAt, time.Millisecond)
	history, err = r.History(ctx, first, 50)
	require.NoError(t, err)
	for _, item := range history {
		require.NotEqual(t, last29.ID, item.ID)
	}
	valid21, err := r.Begin(ctx, first, settings.ModelID, false)
	require.NoError(t, err)
	valid21.Actual, valid21.Verdict, valid21.FinishedAt = &answer21, "pass", &now
	require.NoError(t, r.Finish(ctx, valid21))
	states, err = r.AccountStates(ctx, []int64{first})
	require.NoError(t, err)
	require.Equal(t, 21, *states[0].LastValidAnswer)
	// Exercise the migration against pre-existing valid and failed records.
	failed, err := r.Begin(ctx, first, settings.ModelID, false)
	require.NoError(t, err)
	failed.Verdict, failed.FinishedAt = "inconclusive", &now
	require.NoError(t, r.Finish(ctx, failed))
	_, err = db.ExecContext(ctx, `ALTER TABLE candy_monitor_accounts DROP COLUMN last_valid_answer, DROP COLUMN last_valid_at`)
	require.NoError(t, err)
	migration, err := os.ReadFile("../../migrations/243_candy_monitor_last_valid.sql")
	require.NoError(t, err)
	_, err = db.ExecContext(ctx, string(migration))
	require.NoError(t, err)
	states, err = r.AccountStates(ctx, []int64{first, second})
	require.NoError(t, err)
	require.Equal(t, 21, *states[0].LastValidAnswer)
	require.Nil(t, states[1].LastValidAnswer)
	_, err = db.ExecContext(ctx, `DELETE FROM accounts WHERE id=$1`, first)
	require.NoError(t, err)
	history, err = r.History(ctx, first, 50)
	require.NoError(t, err)
	require.Empty(t, history)
}
