package repository

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"

	"entgo.io/ent/dialect"
	entsql "entgo.io/ent/dialect/sql"
	dbent "github.com/Wei-Shaw/sub2api/ent"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/lib/pq"
	"github.com/stretchr/testify/require"
)

func TestExcelPolicyPostgres(t *testing.T) {
	dsn := os.Getenv("SUB2API_CANDY_TEST_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("disposable PostgreSQL DSN required")
	}
	ctx := context.Background()
	admin, err := sql.Open("postgres", dsn)
	require.NoError(t, err)
	defer admin.Close()
	name := fmt.Sprintf("excel_policy_%d", time.Now().UnixNano())
	_, err = admin.ExecContext(ctx, "CREATE DATABASE "+pq.QuoteIdentifier(name))
	require.NoError(t, err)
	defer func() {
		_, e := admin.ExecContext(ctx, "DROP DATABASE "+pq.QuoteIdentifier(name)+" WITH (FORCE)")
		require.NoError(t, e)
	}()
	db, err := sql.Open("postgres", dsn+" dbname="+name)
	require.NoError(t, err)
	defer db.Close()
	require.NoError(t, ApplyMigrations(ctx, db))
	monitor := NewCandyMonitorRepository(db)
	settings, err := monitor.Settings(ctx)
	require.NoError(t, err)
	settings.AutoExcelOnIncorrect = true
	require.NoError(t, monitor.SaveSettings(ctx, settings))
	loaded, err := monitor.Settings(ctx)
	require.NoError(t, err)
	require.True(t, loaded.AutoExcelOnIncorrect)
	create := func(kind string) int64 {
		var id int64
		require.NoError(t, db.QueryRowContext(ctx, "INSERT INTO accounts(name,platform,type,credentials,extra) VALUES('excel policy','openai',$1,'{}','{}') RETURNING id", kind).Scan(&id))
		return id
	}
	readExtra := func(id int64) map[string]any {
		var raw []byte
		require.NoError(t, db.QueryRowContext(ctx, "SELECT extra FROM accounts WHERE id=$1", id).Scan(&raw))
		var extra map[string]any
		require.NoError(t, json.Unmarshal(raw, &extra))
		return extra
	}
	run := func(id int64, verdict string) {
		result, e := monitor.Begin(ctx, id, service.CandyMonitorDefaultModel, false)
		require.NoError(t, e)
		n := 29
		result.Actual = &n
		result.Verdict = verdict
		now := time.Now()
		result.FinishedAt = &now
		require.NoError(t, monitor.Finish(ctx, result))
	}
	id := create("oauth")
	off := false
	cfg := service.CandyMonitorConfig{Enabled: true, UseDefaults: true, ModelID: service.CandyMonitorDefaultModel, IntervalMinutes: 60, AutoExcelOnIncorrect: &off}
	require.NoError(t, monitor.Configure(ctx, []int64{id}, cfg))
	run(id, "incorrect")
	require.NotEqual(t, true, readExtra(id)["openai_excel_bps"], "account opt-out overrides template")
	cfg.AutoExcelOnIncorrect = nil
	require.NoError(t, monitor.Configure(ctx, []int64{id}, cfg))
	run(id, "invalid_format")
	require.NotEqual(t, true, readExtra(id)["openai_excel_bps"])
	run(id, "incorrect")
	extra := readExtra(id)
	require.Equal(t, true, extra["openai_excel_bps"])
	require.Equal(t, float64(15), extra["base_rpm"])
	require.Equal(t, true, extra["openai_rpm_overflow"])
	require.Equal(t, true, extra["openai_excel_bps_auto_disable_on_403"])
	require.Equal(t, true, extra["openai_excel_bps_cache_creation_as_input"])
	require.Equal(t, "candy_incorrect", extra["openai_excel_bps_last_transition"].(map[string]any)["reason"])
	list, _, err := monitor.List(ctx, service.CandyMonitorFilter{Page: 1, PageSize: 30})
	require.NoError(t, err)
	require.Len(t, list, 1)
	require.Nil(t, list[0].AutoExcelOnIncorrect)
	require.Equal(t, true, list[0].ExcelExtra["openai_excel_bps"])
	states, err := monitor.AccountStates(ctx, []int64{id})
	require.NoError(t, err)
	require.Nil(t, states[0].AutoExcelOnIncorrect)
	client := dbent.NewClient(dbent.Driver(entsql.OpenDB(dialect.Postgres, db)))
	repo := newAccountRepositoryWithSQL(client, db, nil)
	snapshot := &service.Account{ID: id, Platform: service.PlatformOpenAI, Type: service.AccountTypeOAuth, Credentials: map[string]any{}, Extra: extra}
	changed, err := repo.DisableExcelBPSOn403(ctx, snapshot)
	require.NoError(t, err)
	require.True(t, changed)
	extra = readExtra(id)
	require.Equal(t, false, extra["openai_excel_bps"])
	require.Equal(t, "upstream_403", extra["openai_excel_bps_last_transition"].(map[string]any)["reason"])
	run(id, "incorrect")
	require.Equal(t, false, readExtra(id)["openai_excel_bps"], "do not oscillate after a forbidden route")
	require.NoError(t, repo.UpdateExtra(ctx, id, map[string]any{"openai_excel_bps": true}))
	require.Equal(t, "manual", readExtra(id)["openai_excel_bps_last_transition"].(map[string]any)["reason"])
	changed, err = repo.DisableExcelBPSOn403(ctx, snapshot)
	require.NoError(t, err)
	require.False(t, changed, "stale 403 cannot undo a new activation")
	other := create("oauth")
	_, err = repo.BulkUpdate(ctx, []int64{other}, service.AccountBulkUpdate{Extra: map[string]any{"openai_excel_bps": true, "openai_excel_bps_cache_creation_as_input": false}})
	require.NoError(t, err)
	extra = readExtra(other)
	require.Equal(t, float64(15), extra["base_rpm"])
	require.Equal(t, true, extra["openai_excel_bps_auto_disable_on_403"])
	require.NotEqual(t, true, extra["openai_excel_bps_cache_creation_as_input"])

	settings.AutoExcelOnIncorrect = false
	require.NoError(t, monitor.SaveSettings(ctx, settings))
	on := true
	cfg.AutoExcelOnIncorrect = &on
	custom := create("oauth")
	require.NoError(t, monitor.Configure(ctx, []int64{custom}, cfg))
	run(custom, "incorrect")
	require.Equal(t, true, readExtra(custom)["openai_excel_bps"], "account opt-in overrides a disabled template")
	// A manual protocol change made while a probe runs takes precedence.
	pending, e := monitor.Begin(ctx, custom, service.CandyMonitorDefaultModel, false)
	require.NoError(t, e)
	require.NoError(t, repo.UpdateExtra(ctx, custom, map[string]any{"openai_excel_bps": false}))
	wrong := 29
	pending.Actual = &wrong
	pending.Verdict = "incorrect"
	finished := time.Now()
	pending.FinishedAt = &finished
	require.NoError(t, monitor.Finish(ctx, pending))
	require.Equal(t, false, readExtra(custom)["openai_excel_bps"])
	apiKey := create("apikey")
	require.NoError(t, monitor.Configure(ctx, []int64{apiKey}, cfg))
	run(apiKey, "incorrect")
	require.NotEqual(t, true, readExtra(apiKey)["openai_excel_bps"], "do not switch API-key accounts")
}
