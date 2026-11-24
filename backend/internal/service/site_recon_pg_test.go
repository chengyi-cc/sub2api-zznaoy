//go:build unit

package service

import (
	"context"
	"database/sql"
	"os"
	"testing"
	"time"

	_ "github.com/lib/pq"
	"github.com/stretchr/testify/require"
)

// 真实 PostgreSQL 上验证用量日志聚合 SQL 与对账表读写（需要已执行全部迁移的库）。
// 默认跳过；本地验证：SITE_RECON_PG_DSN=postgres://... go test -tags unit -run SiteReconPG ./internal/service/
func TestSiteReconPG_SiteQueriesAndFlow(t *testing.T) {
	dsn := os.Getenv("SITE_RECON_PG_DSN")
	if dsn == "" {
		t.Skip("SITE_RECON_PG_DSN not set")
	}
	db, err := sql.Open("postgres", dsn)
	require.NoError(t, err)
	t.Cleanup(func() { _ = db.Close() })
	ctx := context.Background()
	for _, table := range []string{"site_recon_daily", "site_recon_models", "site_recon_groups", "site_recon_anomalies", "site_recon_anomaly_stats", "site_recon_runs"} {
		_, err := db.Exec("DELETE FROM " + table)
		require.NoError(t, err)
	}

	var userID, keyID, acct1, acct2, groupID int64
	suffix := time.Now().Format("150405.000000")
	require.NoError(t, db.QueryRow(`INSERT INTO users (email, password_hash) VALUES ($1, 'x') RETURNING id`, "recon"+suffix+"@t.local").Scan(&userID))
	require.NoError(t, db.QueryRow(`INSERT INTO api_keys (user_id, key, name) VALUES ($1, $2, 'k') RETURNING id`, userID, "sk-recon-"+suffix).Scan(&keyID))
	require.NoError(t, db.QueryRow(`INSERT INTO accounts (name, platform, type) VALUES ('A', 'openai', 'apikey') RETURNING id`).Scan(&acct1))
	require.NoError(t, db.QueryRow(`INSERT INTO accounts (name, platform, type) VALUES ('B', 'openai', 'apikey') RETURNING id`).Scan(&acct2))
	require.NoError(t, db.QueryRow(`INSERT INTO groups (name) VALUES ($1) RETURNING id`, "VIP"+suffix).Scan(&groupID))

	loc, _ := siteReconLoadLocation("Asia/Shanghai")
	day := "2026-11-09"
	at := func(h, m int) time.Time { return time.Date(2026, 11, 9, h, m, 0, 0, loc) }
	insert := func(acct int64, created time.Time, model, upModel, respModel string, mismatch bool, group *int64, in, out int, total, actual, rate float64, billingType int) {
		_, err := db.Exec(`INSERT INTO usage_logs (user_id, api_key_id, account_id, model, upstream_model, upstream_response_model,
upstream_model_mismatch, group_id, input_tokens, output_tokens, total_cost, actual_cost, rate_multiplier, billing_type, created_at)
VALUES ($1,$2,$3,$4,NULLIF($5,''),NULLIF($6,''),$7,$8,$9,$10,$11,$12,$13,$14,$15)`,
			userID, keyID, acct, model, upModel, respModel, mismatch, group, in, out, total, actual, rate, billingType, created)
		require.NoError(t, err)
	}
	g := &groupID
	insert(acct1, at(0, 5), "gpt-x", "gpt-x-up", "", false, g, 10, 10, 1.0, 0.8, 0.8, 0)                  // 正常
	insert(acct1, at(12, 0), "gpt-x", "gpt-x-up", "", false, g, 10, 10, 2.0, 1.6, 0.8, 0)                 // 正常
	insert(acct1, at(13, 0), "gpt-x", "", "", false, g, 10, 10, 1.0, 0, 0.8, 0)                           // 漏扣
	insert(acct1, at(14, 0), "free-m", "", "", false, nil, 10, 10, 0, 0, 1, 0)                            // 未定价
	insert(acct1, at(15, 0), "gpt-x", "", "", false, g, 10, 10, 1.0, 0, 0, 1)                             // 倍率 0（订阅）
	insert(acct1, at(16, 0), "gpt-x", "", "gpt-x-mini", true, g, 10, 10, 1.0, 0.8, 0.8, 0)                // 模型不一致
	insert(acct1, at(23, 59), "gpt-x", "", "", false, g, 1, 1, 0.5, 0.4, 0.8, 1)                          // 订阅扣费不算收入
	insert(acct1, time.Date(2026, 11, 8, 23, 30, 0, 0, loc), "gpt-x", "", "", false, g, 1, 1, 9, 9, 1, 0) // 前一天（本地时区）
	insert(acct2, at(10, 0), "gpt-y", "", "", false, g, 1, 1, 3, 3, 1, 0)

	site := newSiteReconPGSite(db, nil, "Asia/Shanghai", loc)
	win, err := site.WindowRequests(ctx, day, day)
	require.NoError(t, err)
	require.Equal(t, int64(7), win[acct1])
	require.Equal(t, int64(1), win[acct2])

	stats, err := site.DaysStats(ctx, []int64{acct1}, []string{day, "2026-11-08"})
	require.NoError(t, err)
	require.Equal(t, int64(7), stats[day].Requests)
	require.InDelta(t, 6.5, stats[day].ListCost, 1e-9)
	require.InDelta(t, 0.8+1.6+0.8, stats[day].Revenue, 1e-9)
	require.Equal(t, int64(1), stats["2026-11-08"].Requests)

	scan, err := site.ScanDay(ctx, []int64{acct1}, day)
	require.NoError(t, err)
	require.Equal(t, int64(7), scan.N)
	kinds := map[string]int64{}
	for k, v := range scan.Stats {
		kinds[k.Kind] += v.Requests
	}
	require.Equal(t, map[string]int64{SiteReconKindZeroPaid: 1, SiteReconKindUnpriced: 1, SiteReconKindFree: 1, SiteReconKindMismatch: 1}, kinds)
	require.Contains(t, scan.Stats, siteReconStatKey{Kind: SiteReconKindMismatch, Model: "gpt-x", Detail: "gpt-x-mini"})
	require.Len(t, scan.Samples, 3) // 模型不一致不留样本
	models := scan.Models()
	require.Equal(t, int64(2), models["gpt-x-up"].Requests)
	require.Contains(t, models, "free-m")
	require.Contains(t, scan.GroupModels, siteReconGroupModelKey{Group: "#-", Rate: 1, Model: "free-m"})

	// 整体流程：真实 PG 的本站数据 + 对账表，上游用假的。
	store := newSiteReconStore(db, true)
	pgSite := &siteReconPGTestSite{siteReconPGSite: site, accounts: []siteReconAccount{
		{ID: acct1, Name: "A", Status: StatusActive, BaseURL: "https://up.example/v1", APIKey: "ka"},
		{ID: acct2, Name: "B", Status: StatusActive, BaseURL: "https://up.example/v1", APIKey: "kb"},
	}}
	e := &siteReconEngine{store: store, site: pgSite, http: newTestSiteReconHTTP(), cfg: defaultSiteReconConfig(), loc: loc,
		now: func() time.Time { return time.Date(2026, 11, 10, 9, 0, 0, 0, loc) }, log: func(string) {},
		openUpstream: func(_ context.Context, u *siteReconUnit) (siteReconUpstream, string, string) {
			costs := map[string][2]float64{day: {7, 2}}
			if u.APIKey == "kb" {
				costs = map[string][2]float64{day: {1, 1}}
			}
			return &fakeSiteReconUpstream{costs: costs}, SiteReconStatusOK, ""
		}}
	summary, err := e.run(ctx, siteReconRunOptions{Target: day, Days: 2})
	require.NoError(t, err)
	require.Equal(t, 2, summary.Units)

	totals, err := store.dayTotals(ctx, day)
	require.NoError(t, err)
	require.InDelta(t, 3.2+3, totals.Rev, 1e-9)
	require.InDelta(t, 3.0, totals.Cost, 1e-9)
	require.Equal(t, int64(2), totals.Scanned)
	groups, err := store.groupPnL(ctx, day)
	require.NoError(t, err)
	sum := 0.0
	for _, gp := range groups {
		if gp.Cost != nil {
			sum += *gp.Cost
		}
	}
	require.InDelta(t, 3.0, sum, 1e-9)
	_, err = store.alerts(ctx, day, 2)
	require.NoError(t, err)
	_, err = store.history(ctx, 60)
	require.NoError(t, err)
	_, err = store.anomalyHistory(ctx, 14)
	require.NoError(t, err)
	_, err = store.modelReport(ctx, day)
	require.NoError(t, err)
	samples, err := store.anomalySamples(ctx, day)
	require.NoError(t, err)
	require.Len(t, samples, 3)

	id, err := store.insertRun(ctx, "manual", day, 2, time.Now())
	require.NoError(t, err)
	require.Positive(t, id)
	require.NoError(t, store.finishRun(ctx, id, day, 2, SiteReconRunOK, `{"target":"2026-11-09"}`, "log", time.Now()))
	runs, err := store.listRuns(ctx, 5)
	require.NoError(t, err)
	require.NotEmpty(t, runs)
	require.NoError(t, store.upsertToken(ctx, "up.example", "cipher", "****abcd", 1, time.Now()))
	require.NoError(t, store.upsertToken(ctx, "up.example", "cipher2", "****efgh", 1, time.Now()))
	ok, err := store.deleteToken(ctx, "up.example")
	require.NoError(t, err)
	require.True(t, ok)

	// 自有持久密钥写入真实的 security_secrets 表，换实例后仍能解开。
	svc := newSiteReconService(db, true, nil, &upstreamBillingProbeSettingRepo{}, nil, nil)
	t.Cleanup(svc.Stop)
	require.NoError(t, svc.SetToken(ctx, "pg.example.com", "pg-access-token-1234", 1))
	again := newSiteReconService(db, true, nil, &upstreamBillingProbeSettingRepo{}, nil, nil)
	t.Cleanup(again.Stop)
	require.Equal(t, "pg-access-token-1234", again.decryptTokens(ctx, func(string) {})["pg.example.com"])
	_, err = db.Exec(`DELETE FROM site_recon_upstream_tokens WHERE host = 'pg.example.com'`)
	require.NoError(t, err)
}

type siteReconPGTestSite struct {
	*siteReconPGSite
	accounts []siteReconAccount
}

func (s *siteReconPGTestSite) ListAPIKeyAccounts(context.Context) ([]siteReconAccount, error) {
	return s.accounts, nil
}
