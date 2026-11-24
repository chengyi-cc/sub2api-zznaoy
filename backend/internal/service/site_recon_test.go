//go:build unit

package service

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

// 迁移自原「站点对账」Python 服务的 tests/test_recon.py，口径保持一致。

func srOurs(req int64, rev float64) siteReconDayStats {
	return siteReconDayStats{Requests: req, Revenue: rev, ListCost: rev}
}

func srUp(req int64, cost float64) *siteReconUpDay {
	return &siteReconUpDay{Requests: req, Cost: cost, ListCost: siteReconF64(cost * 2)}
}

func TestSiteReconJudgeDay(t *testing.T) {
	j := siteReconJudgeDay(srOurs(10, 5), srUp(10, 2), "")
	require.Equal(t, 2.0, *j.Cost)
	require.Equal(t, int64(10), *j.UpRequests)
	require.Nil(t, j.Error)

	idle := siteReconJudgeDay(srOurs(0, 0), srUp(0, 0), "")
	require.Equal(t, 0.0, *idle.Cost)
	require.Nil(t, idle.Error)

	// 账号换过 key：本站收了钱，上游这把 key 当天没记录。成本是“不知道”，不能当 0 算成纯利润。
	swapped := siteReconJudgeDay(srOurs(4921, 98.23), srUp(0, 0), "")
	require.Nil(t, swapped.Cost)
	require.Contains(t, *swapped.Error, "换过 key")

	failed := siteReconJudgeDay(srOurs(3, 1.5), nil, "上游取数失败: boom")
	require.Equal(t, 1.5, failed.Revenue)
	require.Nil(t, failed.Cost)
	require.Equal(t, "上游取数失败: boom", *failed.Error)

	outOfRange := siteReconJudgeDay(srOurs(3, 1.5), nil, "")
	require.Nil(t, outOfRange.Cost)
	require.Contains(t, *outOfRange.Error, "范围")
}

func TestSiteReconKeepKnown(t *testing.T) {
	old := &SiteReconDaily{Cost: siteReconF64(2.99), UpRequests: siteReconI64(481), UpListCost: siteReconF64(20)}
	for _, next := range []siteReconJudged{
		siteReconJudgeDay(srOurs(480, 5.37), srUp(364, 2.28), ""), // 上游清理了旧数据
		siteReconJudgeDay(srOurs(480, 5.37), srUp(0, 0), ""),      // 上游完全查不到了
		siteReconJudgeDay(srOurs(480, 5.37), nil, "上游取数失败: 502"),  // 上游故障 / key 失效
	} {
		j := siteReconKeepKnown(old, next)
		require.Equal(t, 2.99, *j.Cost)
		require.Equal(t, int64(481), *j.UpRequests)
		require.Nil(t, j.Error)
		require.True(t, j.Kept)
		require.Equal(t, 5.37, j.Revenue)
	}

	more := siteReconKeepKnown(old, siteReconJudgeDay(srOurs(480, 5.37), srUp(485, 3.05), ""))
	require.Equal(t, 3.05, *more.Cost)
	require.False(t, more.Kept)
	refund := siteReconKeepKnown(old, siteReconJudgeDay(srOurs(480, 5.37), srUp(481, 2.50), ""))
	require.Equal(t, 2.50, *refund.Cost) // 请求数没变、上游退了钱：认新的

	unknown := siteReconJudgeDay(srOurs(10, 5), nil, "上游取数失败: x")
	require.Equal(t, unknown, siteReconKeepKnown(nil, unknown))
	require.Equal(t, unknown, siteReconKeepKnown(&SiteReconDaily{}, unknown))

	// 以前这天两边都是 0（成本 0），后来本站补记出收入而上游查不到：不能沿用成本 0 把它当纯利润。
	j := siteReconKeepKnown(&SiteReconDaily{Cost: siteReconF64(0), UpRequests: siteReconI64(0), UpListCost: siteReconF64(0)},
		siteReconJudgeDay(srOurs(10, 5), srUp(0, 0), ""))
	require.Nil(t, j.Cost)
}

func TestSiteReconAllocateGroupCost(t *testing.T) {
	gm := map[siteReconGroupModelKey]*siteReconAgg{
		{"A", 1.0, "x"}: {Requests: 1, ListCost: 30, Revenue: 30},
		{"B", 1.2, "x"}: {Requests: 1, ListCost: 10, Revenue: 12},
		{"B", 1.2, "y"}: {Requests: 1, ListCost: 60, Revenue: 72},
	}
	om := map[string]*siteReconAgg{"x": {ListCost: 40}, "y": {ListCost: 60}}
	um := map[string]siteReconUpModel{"x": {Cost: 20}, "z-renamed": {Cost: 30}} // y 在上游叫别的名字
	out := siteReconAllocateGroupCost(gm, om, um, siteReconF64(50))
	sum := 0.0
	for _, v := range out {
		sum += v
	}
	require.InDelta(t, 50.0, sum, 1e-9)
	require.InDelta(t, 15.0+30.0*0.3, out[siteReconGroupKey{"A", 1.0}], 1e-9) // x 的 3/4，加上剩余按原价占比

	require.Empty(t, siteReconAllocateGroupCost(gm, om, um, nil))
}

func TestSiteReconSchedule(t *testing.T) {
	bj := time.FixedZone("BJ", 8*3600)
	at := func(h, m int) time.Time { return time.Date(2026, 11, 10, h, m, 0, 0, bj) }
	require.Equal(t, "", siteReconDue(at(7, 59), true, "08:00", SiteReconScheduleState{}))
	require.Equal(t, "schedule", siteReconDue(at(8, 0), true, "08:00", SiteReconScheduleState{}))
	require.Equal(t, "schedule", siteReconDue(at(15, 30), true, "08:00", SiteReconScheduleState{LastDay: "2026-11-09"}))
	require.Equal(t, "", siteReconDue(at(15, 30), true, "08:00", SiteReconScheduleState{LastDay: "2026-11-10"}))
	require.Equal(t, "", siteReconDue(at(9, 0), false, "08:00", SiteReconScheduleState{}))

	state := SiteReconScheduleState{LastDay: "2026-11-10"}
	for n := 1; n <= siteReconMaxRetries; n++ {
		state = siteReconAfterScheduledRun(at(8, 5), state, false)
		require.Equal(t, n, state.Retries)
		require.Equal(t, "", siteReconDue(at(8, 20), true, "08:00", state))
		require.Equal(t, "retry", siteReconDue(at(8, 40), true, "08:00", state))
	}
	state = siteReconAfterScheduledRun(at(8, 5), state, false)
	require.Equal(t, "", state.RetryAt) // 重试用完，等明天
	require.Equal(t, "", siteReconDue(at(23, 0), true, "08:00", state))

	ok := siteReconAfterScheduledRun(at(8, 5), SiteReconScheduleState{LastDay: "2026-11-10", Retries: 2, RetryAt: "2026-11-10T08:35:00+08:00"}, true)
	require.Equal(t, 0, ok.Retries)
	require.Equal(t, "", ok.RetryAt)

	require.Equal(t, "2026-11-10 08:00", siteReconNextRun(at(7, 0), true, "08:00", SiteReconScheduleState{}))
	require.Equal(t, "2026-11-11 08:00", siteReconNextRun(at(9, 0), true, "08:00", SiteReconScheduleState{LastDay: "2026-11-10"}))
	require.Equal(t, "", siteReconNextRun(at(9, 0), false, "08:00", SiteReconScheduleState{}))
}

func TestSiteReconConfigNormalize(t *testing.T) {
	c, err := normalizeSiteReconConfig(SiteReconConfig{ScheduleTime: "8:05", Days: 999, Exclude: []int64{7, 7, 3, -1}, BalanceDays: -1})
	require.NoError(t, err)
	require.Equal(t, "Asia/Shanghai", c.Timezone)
	require.Equal(t, "08:05", c.ScheduleTime)
	require.Equal(t, 90, c.Days)
	require.Equal(t, []int64{3, 7}, c.Exclude)
	require.Equal(t, 0.0, c.BalanceDays)
	_, err = normalizeSiteReconConfig(SiteReconConfig{ScheduleTime: "25:99"})
	require.Error(t, err)
	_, err = normalizeSiteReconConfig(SiteReconConfig{Timezone: "Mars/Base"})
	require.Error(t, err)
	require.Equal(t, "up.com:8080", siteReconCleanHost(" https://Up.com:8080/path?x=1 "))
}

// ---------------------------------------------------------------- HTTP

func newTestSiteReconHTTP() *siteReconHTTP {
	h := newSiteReconHTTP()
	h.sleep = func(context.Context, time.Duration) error { return nil }
	return h
}

func TestSiteReconHTTPRetryAndErrors(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch calls.Add(1) {
		case 1:
			w.WriteHeader(502)
		default:
			_, _ = w.Write([]byte(`{"ok":1}`))
		}
	}))
	defer srv.Close()
	h := newTestSiteReconHTTP()
	var out map[string]int
	require.NoError(t, h.getJSON(context.Background(), srv.URL+"/y?k=secret", "", nil, 5*time.Second, 3, &out))
	require.Equal(t, 1, out["ok"])
	require.Equal(t, int32(2), calls.Load())

	calls.Store(0)
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(401)
		_, _ = w.Write([]byte("INVALID_API_KEY"))
	}))
	defer bad.Close()
	err := h.getJSON(context.Background(), bad.URL+"/y?k=secret", "", nil, 5*time.Second, 3, &out)
	require.Error(t, err)
	require.Equal(t, int32(1), calls.Load()) // 4xx 立即报错
	require.Contains(t, err.Error(), "INVALID_API_KEY")
	require.NotContains(t, err.Error(), "secret")

	calls.Store(0)
	down := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(503)
	}))
	defer down.Close()
	require.Error(t, h.getJSON(context.Background(), down.URL+"/y", "", nil, 5*time.Second, 3, &out))
	require.Equal(t, int32(4), calls.Load())
}

// 带令牌的请求遇到跳转不跟随：令牌不会被带到别的主机。
func TestSiteReconHTTPDoesNotFollowRedirects(t *testing.T) {
	var leaked atomic.Bool
	evil := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			leaked.Store(true)
		}
		_, _ = w.Write([]byte(`{}`))
	}))
	defer evil.Close()
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, evil.URL+"/steal", http.StatusFound)
	}))
	defer up.Close()
	var out map[string]any
	err := newTestSiteReconHTTP().getJSON(context.Background(), up.URL+"/api/log/self", "", map[string]string{"Authorization": "Bearer tok-123456"}, 5*time.Second, 0, &out)
	require.Error(t, err)
	require.Contains(t, err.Error(), "HTTP 302")
	require.False(t, leaked.Load())
}

func TestSiteReconSub2APIDailyMissingDaysAreZero(t *testing.T) {
	loc := time.FixedZone("BJ", 8*3600)
	now := time.Date(2026, 11, 10, 9, 0, 0, 0, loc)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "Bearer k", r.Header.Get("Authorization"))
		require.Equal(t, "/v1/usage", r.URL.Path)
		_, _ = w.Write([]byte(`{"balance":9.5,"daily_usage":[{"date":"2026-11-09","requests":3,"cost":2,"actual_cost":1}]}`))
	}))
	defer srv.Close()
	u := &siteReconSub2APIUpstream{h: newTestSiteReconHTTP(), origin: srv.URL, apiKey: "k", tz: "Asia/Shanghai", loc: loc, now: func() time.Time { return now }}
	out, bal, err := u.Daily(context.Background(), []string{"2026-11-09", "2026-11-08"})
	require.NoError(t, err)
	require.Equal(t, 1.0, out["2026-11-09"].Cost)
	require.Equal(t, int64(0), out["2026-11-08"].Requests)
	require.Equal(t, 9.5, *bal)
}

// NewAPI：账号 key 只发 /api/usage/token/，访问令牌只发日志接口；倍率从 other.group_ratio 还原原价。
func TestSiteReconNewAPIUpstream(t *testing.T) {
	loc := time.FixedZone("BJ", 8*3600)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		switch r.URL.Path {
		case "/api/usage/token/":
			require.Equal(t, "Bearer sk-acct", auth)
			_, _ = w.Write([]byte(`{"code":true,"data":{"name":"relay-key"}}`))
		case "/api/log/self/stat":
			require.Equal(t, "Bearer access-tok", auth)
			require.Equal(t, "relay-key", r.URL.Query().Get("token_name"))
			_, _ = w.Write([]byte(`{"success":true,"data":{"quota":1000000}}`))
		case "/api/log/self":
			require.Equal(t, "Bearer access-tok", auth)
			if r.URL.Query().Get("page_size") == "1" {
				_, _ = w.Write([]byte(`{"success":true,"data":{"total":2,"items":[]}}`))
				return
			}
			_, _ = w.Write([]byte(`{"success":true,"data":{"total":2,"items":[
				{"model_name":"m","quota":500000,"other":"{\"group_ratio\":0.5}"},
				{"model_name":"m","quota":500000,"other":""}]}}`))
		case "/api/user/self":
			_, _ = w.Write([]byte(`{"success":true,"data":{"quota":2500000}}`))
		default:
			w.WriteHeader(404)
		}
	}))
	defer srv.Close()
	h := newTestSiteReconHTTP()
	u, err := newSiteReconNewAPIUpstream(context.Background(), h, srv.URL, "sk-acct", "access-tok", "", loc, map[string]any{"quota_per_unit": 500000.0})
	require.NoError(t, err)
	daily, bal, err := u.Daily(context.Background(), []string{"2026-11-09"})
	require.NoError(t, err)
	require.Equal(t, int64(2), daily["2026-11-09"].Requests)
	require.Equal(t, 2.0, daily["2026-11-09"].Cost)
	require.Nil(t, daily["2026-11-09"].ListCost)
	require.Equal(t, 5.0, *bal)
	models, err := u.DayModels(context.Background(), "2026-11-09")
	require.NoError(t, err)
	require.Equal(t, int64(2), models["m"].Requests)
	require.InDelta(t, 2.0, models["m"].Cost, 1e-9)
	require.InDelta(t, 3.0, models["m"].ListCost, 1e-9) // 1/0.5 + 1
}

func TestSiteReconClassifyAndRedact(t *testing.T) {
	st, _ := siteReconClassifyUpstreamError("HTTP 401 https://x/v1/usage: INVALID_API_KEY")
	require.Equal(t, SiteReconStatusDeadKey, st)
	st, _ = siteReconClassifyUpstreamError("HTTP 401 https://x/api/usage/token/: no")
	require.Equal(t, SiteReconStatusDeadKey, st)
	st, _ = siteReconClassifyUpstreamError("HTTP 502 https://x/v1/usage: bad gateway")
	require.Equal(t, SiteReconStatusUpstreamError, st)
	require.Equal(t, "echo *** here", siteReconRedact("echo sk-secret-value here", "sk-secret-value"))
}

// ---------------------------------------------------------------- 整体流程（假本站 + 假上游 + SQLite）

type fakeSiteReconAccount struct {
	name   string
	status string
	key    string
	usage  map[string][2]float64 // day -> (requests, revenue)
}

type fakeSiteReconSite struct {
	world map[int64]*fakeSiteReconAccount
}

func (f *fakeSiteReconSite) ListAPIKeyAccounts(context.Context) ([]siteReconAccount, error) {
	var out []siteReconAccount
	for id := int64(1); id <= 20; id++ {
		a := f.world[id]
		if a == nil {
			continue
		}
		status := a.status
		if status == "" {
			status = StatusActive
		}
		out = append(out, siteReconAccount{ID: id, Name: a.name, Status: status, BaseURL: "https://up.example/v1", APIKey: a.key})
	}
	return out, nil
}

func (f *fakeSiteReconSite) WindowRequests(_ context.Context, first, last string) (map[int64]int64, error) {
	out := map[int64]int64{}
	for id, a := range f.world {
		for d, u := range a.usage {
			if d >= first && d <= last {
				out[id] += int64(u[0])
			}
		}
	}
	return out, nil
}

func (f *fakeSiteReconSite) DaysStats(_ context.Context, ids []int64, days []string) (map[string]siteReconDayStats, error) {
	out := map[string]siteReconDayStats{}
	for _, d := range days {
		var st siteReconDayStats
		for _, id := range ids {
			u := f.world[id].usage[d]
			st.Requests += int64(u[0])
			st.Revenue += u[1]
			st.ListCost += u[1]
		}
		out[d] = st
	}
	return out, nil
}

func (f *fakeSiteReconSite) ScanDay(_ context.Context, ids []int64, day string) (*siteReconScan, error) {
	res := &siteReconScan{Stats: map[siteReconStatKey]*siteReconStat{}, GroupModels: map[siteReconGroupModelKey]*siteReconAgg{}}
	for _, id := range ids {
		u := f.world[id].usage[day]
		if u[0] == 0 {
			continue
		}
		k := siteReconGroupModelKey{Group: "G", Rate: 1, Model: "m"}
		a := res.GroupModels[k]
		if a == nil {
			a = &siteReconAgg{}
			res.GroupModels[k] = a
		}
		a.Requests += int64(u[0])
		a.ListCost += u[1]
		a.Revenue += u[1]
		res.N += int64(u[0])
	}
	return res, nil
}

type fakeSiteReconUpstream struct {
	costs map[string][2]float64 // day -> (requests, cost)
}

func (u *fakeSiteReconUpstream) Describe() string { return "fake" }

func (u *fakeSiteReconUpstream) Daily(_ context.Context, days []string) (map[string]siteReconUpDay, *float64, error) {
	out := map[string]siteReconUpDay{}
	for _, d := range days {
		c := u.costs[d]
		out[d] = siteReconUpDay{Requests: int64(c[0]), Cost: c[1], ListCost: siteReconF64(c[1] * 2)}
	}
	return out, siteReconF64(100), nil
}

func (u *fakeSiteReconUpstream) DayModels(_ context.Context, day string) (map[string]siteReconUpModel, error) {
	c := u.costs[day]
	if c[0] == 0 {
		return map[string]siteReconUpModel{}, nil
	}
	return map[string]siteReconUpModel{"m": {Requests: int64(c[0]), ListCost: c[1] * 2, Cost: c[1]}}, nil
}

func newSiteReconTestDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", "file:"+strings.ReplaceAll(t.Name(), "/", "_")+"?mode=memory&cache=shared")
	require.NoError(t, err)
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { _ = db.Close() })
	_, err = db.Exec(siteReconSQLiteDDL)
	require.NoError(t, err)
	return db
}

type siteReconFlow struct {
	t       *testing.T
	store   *siteReconStore
	site    *fakeSiteReconSite
	costs   map[string]map[string][2]float64 // key -> day -> (requests, cost)
	open    func(ctx context.Context, u *siteReconUnit) (siteReconUpstream, string, string)
	exclude []int64
}

const (
	srD1 = "2026-11-07"
	srD2 = "2026-11-08"
	srD3 = "2026-11-09"
)

func newSiteReconFlow(t *testing.T) *siteReconFlow {
	f := &siteReconFlow{t: t, store: newSiteReconStore(newSiteReconTestDB(t), false), site: &fakeSiteReconSite{world: map[int64]*fakeSiteReconAccount{}},
		costs: map[string]map[string][2]float64{}}
	f.open = func(_ context.Context, u *siteReconUnit) (siteReconUpstream, string, string) {
		return &fakeSiteReconUpstream{costs: f.costs[u.APIKey]}, SiteReconStatusOK, ""
	}
	return f
}

func (f *siteReconFlow) run(target string) *siteReconSummary {
	cfg := defaultSiteReconConfig()
	cfg.Exclude = f.exclude
	loc := time.FixedZone("BJ", 8*3600)
	e := &siteReconEngine{store: f.store, site: f.site, http: newTestSiteReconHTTP(), cfg: cfg, loc: loc,
		now: func() time.Time { return time.Date(2026, 11, 12, 9, 0, 0, 0, loc) }, log: func(string) {}, openUpstream: f.open}
	s, err := e.run(context.Background(), siteReconRunOptions{Target: target, Days: 3})
	require.NoError(f.t, err)
	return s
}

func (f *siteReconFlow) total(day string) (float64, float64, int) {
	var rev, cost sql.NullFloat64
	var n int
	require.NoError(f.t, f.store.db.QueryRow(`SELECT SUM(revenue), SUM(cost), COUNT(*) FROM site_recon_daily WHERE day = ?`, day).Scan(&rev, &cost, &n))
	return math.Round(rev.Float64*100) / 100, math.Round(cost.Float64*100) / 100, n
}

func TestSiteReconFlow_DisabledAndDeletedAccountsKeepHistory(t *testing.T) {
	f := newSiteReconFlow(t)
	f.site.world[1] = &fakeSiteReconAccount{name: "A", key: "ka", usage: map[string][2]float64{srD1: {10, 100}, srD2: {10, 100}}}
	f.site.world[2] = &fakeSiteReconAccount{name: "B", key: "kb", usage: map[string][2]float64{srD1: {5, 50}}}
	f.costs = map[string]map[string][2]float64{"ka": {srD1: {10, 40}, srD2: {10, 40}}, "kb": {srD1: {5, 20}}}
	f.run(srD2)
	rev, cost, n := f.total(srD1)
	require.Equal(t, []any{150.0, 60.0, 2}, []any{rev, cost, n})

	f.site.world[2].status = "error" // B 报错停用了，但 D1 的钱是真的
	f.run(srD3)
	rev, cost, n = f.total(srD1)
	require.Equal(t, []any{150.0, 60.0, 2}, []any{rev, cost, n})

	delete(f.site.world, 2) // B 被从本站删除：历史原样保留
	f.run(srD3)
	rev, cost, n = f.total(srD1)
	require.Equal(t, []any{150.0, 60.0, 2}, []any{rev, cost, n})
}

func TestSiteReconFlow_SharedKeyCountedOnce(t *testing.T) {
	f := newSiteReconFlow(t)
	f.site.world[1] = &fakeSiteReconAccount{name: "A", key: "shared", usage: map[string][2]float64{srD1: {10, 100}}}
	f.site.world[2] = &fakeSiteReconAccount{name: "B", key: "shared", usage: map[string][2]float64{srD1: {10, 80}, srD2: {10, 80}, srD3: {10, 80}}}
	f.costs = map[string]map[string][2]float64{"shared": {srD1: {20, 90}, srD2: {10, 45}, srD3: {10, 45}}}
	f.run(srD1)
	rev, cost, n := f.total(srD1)
	require.Equal(t, []any{180.0, 90.0, 1}, []any{rev, cost, n}) // 两个账号合成一行，成本只算一次
	var primary int64
	require.NoError(t, f.store.db.QueryRow(`SELECT account_id FROM site_recon_daily WHERE day = ?`, srD1).Scan(&primary))

	f.run("2026-11-11") // 窗口往后移到 A 没有消费的日子
	rev, cost, n = f.total(srD3)
	require.Equal(t, []any{80.0, 45.0, 1}, []any{rev, cost, n})
	rev, cost, n = f.total(srD1)
	require.Equal(t, []any{180.0, 90.0, 1}, []any{rev, cost, n}) // 窗口外的历史不动
	var again int64
	require.NoError(t, f.store.db.QueryRow(`SELECT account_id FROM site_recon_daily WHERE day = ?`, srD1).Scan(&again))
	require.Equal(t, primary, again)
}

func TestSiteReconFlow_IdleAndExcludedRowsDropped(t *testing.T) {
	f := newSiteReconFlow(t)
	f.site.world[1] = &fakeSiteReconAccount{name: "A", key: "ka", usage: map[string][2]float64{srD1: {10, 100}, srD3: {1, 1}}}
	f.site.world[2] = &fakeSiteReconAccount{name: "B", key: "kb", usage: map[string][2]float64{srD1: {5, 50}, srD3: {1, 1}}}
	f.costs = map[string]map[string][2]float64{"ka": {srD1: {10, 40}}, "kb": {srD1: {5, 20}}}
	f.run(srD3)
	f.exclude = []int64{2}
	f.run(srD3)
	rev, cost, n := f.total(srD1)
	require.Equal(t, []any{100.0, 40.0, 1}, []any{rev, cost, n})
}

func TestSiteReconFlow_ScanWritesGroupsAndBackfills(t *testing.T) {
	f := newSiteReconFlow(t)
	f.site.world[1] = &fakeSiteReconAccount{name: "A", key: "ka", usage: map[string][2]float64{srD1: {4, 40}, srD2: {2, 20}, srD3: {2, 30}}}
	f.costs = map[string]map[string][2]float64{"ka": {srD1: {4, 10}, srD2: {2, 5}, srD3: {2, 6}}}
	f.run(srD3) // 只跑了 D3，但 D1、D2 没扫过，要顺带补上
	got := map[string]float64{}
	rows, err := f.store.db.Query(`SELECT day, SUM(cost) FROM site_recon_groups GROUP BY day`)
	require.NoError(t, err)
	for rows.Next() {
		var d string
		var c float64
		require.NoError(t, rows.Scan(&d, &c))
		got[d] = math.Round(c*100) / 100
	}
	_ = rows.Close()
	require.Equal(t, map[string]float64{srD1: 10, srD2: 5, srD3: 6}, got)
	scanned := map[string]int64{}
	rows, err = f.store.db.Query(`SELECT day, scanned_rows FROM site_recon_daily`)
	require.NoError(t, err)
	for rows.Next() {
		var d string
		var n int64
		require.NoError(t, rows.Scan(&d, &n))
		scanned[d] = n
	}
	_ = rows.Close()
	require.Equal(t, map[string]int64{srD1: 4, srD2: 2, srD3: 2}, scanned)
}

func TestSiteReconFlow_UpstreamOutageKeepsHistory(t *testing.T) {
	f := newSiteReconFlow(t)
	f.site.world[1] = &fakeSiteReconAccount{name: "A", key: "ka", usage: map[string][2]float64{srD1: {10, 100}, srD2: {10, 100}, srD3: {10, 100}}}
	f.costs = map[string]map[string][2]float64{"ka": {srD1: {10, 40}, srD2: {10, 40}}}
	f.run(srD2)
	f.open = func(context.Context, *siteReconUnit) (siteReconUpstream, string, string) {
		return nil, SiteReconStatusDeadKey, "上游取数失败: INVALID_API_KEY"
	}
	summary := f.run(srD3) // key 失效后的第一次对账
	rev, cost, n := f.total(srD1)
	require.Equal(t, []any{100.0, 40.0, 1}, []any{rev, cost, n}) // 已对好的日子不动
	rev, cost, n = f.total(srD2)
	require.Equal(t, []any{100.0, 40.0, 1}, []any{rev, cost, n})
	rev, cost, n = f.total(srD3)
	require.Equal(t, []any{100.0, 0.0, 1}, []any{rev, cost, n}) // 新的一天拿不到成本
	p3, err := f.store.pending(context.Background(), srD3)
	require.NoError(t, err)
	require.Len(t, p3, 1)
	p2, err := f.store.pending(context.Background(), srD2)
	require.NoError(t, err)
	require.Empty(t, p2)
	require.Contains(t, summary.Problems, SiteReconStatusDeadKey)
	var nullCost int
	require.NoError(t, f.store.db.QueryRow(`SELECT COUNT(*) FROM site_recon_groups WHERE day = ? AND cost IS NULL`, srD1).Scan(&nullCost))
	require.Equal(t, 0, nullCost)
}

func TestSiteReconFlow_KeySwappedDayGoesToPending(t *testing.T) {
	f := newSiteReconFlow(t)
	f.site.world[1] = &fakeSiteReconAccount{name: "A", key: "new-key", usage: map[string][2]float64{srD3: {100, 98}}}
	f.run(srD3) // 新 key 在上游没有任何用量
	ctx := context.Background()
	tt, err := f.store.dayTotals(ctx, srD3)
	require.NoError(t, err)
	require.Equal(t, 0.0, tt.Rev)
	require.Equal(t, 0.0, tt.Cost)
	require.InDelta(t, 98.0, tt.Missing, 1e-9)
	p, err := f.store.pending(ctx, srD3)
	require.NoError(t, err)
	require.Len(t, p, 1)
	alerts, err := f.store.alerts(ctx, srD3, 2)
	require.NoError(t, err)
	found := false
	for _, a := range alerts {
		found = found || a.Level == "warn"
	}
	require.True(t, found)
}

// ---------------------------------------------------------------- 余额预警

func addSiteReconWallet(t *testing.T, s *siteReconStore, id int64, host string, balance, cost float64) {
	for i := 0; i < 3; i++ {
		d := &SiteReconDaily{Day: siteReconShiftDay(srD3, -i), AccountID: id, AccountName: fmt.Sprintf("acct%d", id), Upstream: host,
			Cost: siteReconF64(cost), Revenue: cost * 2, OurRequests: 1, UpRequests: siteReconI64(1)}
		if i == 0 {
			d.UpBalance = siteReconF64(balance)
		}
		require.NoError(t, s.upsertDaily(context.Background(), d))
	}
}

func TestSiteReconWallets(t *testing.T) {
	s := newSiteReconStore(newSiteReconTestDB(t), false)
	addSiteReconWallet(t, s, 1, "a.com", 672.59, 380) // 与 2 共用钱包：日耗合并
	addSiteReconWallet(t, s, 2, "a.com", 672.50, 20)
	addSiteReconWallet(t, s, 3, "b.com", -0.01, 24) // 用完了还在扣
	addSiteReconWallet(t, s, 4, "c.com", 5000, 10)  // 充足
	addSiteReconWallet(t, s, 5, "d.com", 0, 0)      // 没余额也没消耗：不用管
	ws, err := s.wallets(context.Background(), srD3, 2)
	require.NoError(t, err)
	by := map[string]*SiteReconWallet{}
	for _, w := range ws {
		by[w.Host] = w
	}
	require.Equal(t, "low", by["a.com"].Level)
	require.InDelta(t, 400.0, by["a.com"].Burn, 1e-9)
	require.Len(t, by["a.com"].Names, 2)
	require.Equal(t, "empty", by["b.com"].Level)
	require.Equal(t, "ok", by["c.com"].Level)
	require.Equal(t, "idle", by["d.com"].Level)
	alerts, err := s.alerts(context.Background(), srD3, 2)
	require.NoError(t, err)
	bad := 0
	for _, a := range alerts {
		if a.Level == "bad" {
			bad++
		}
	}
	require.Equal(t, 2, bad)
}

func TestSiteReconWalletsDifferentBalancesAreSeparate(t *testing.T) {
	s := newSiteReconStore(newSiteReconTestDB(t), false)
	addSiteReconWallet(t, s, 1, "a.com", 500, 10)
	addSiteReconWallet(t, s, 2, "a.com", 20, 30)
	ws, err := s.wallets(context.Background(), srD3, 2)
	require.NoError(t, err)
	require.Len(t, ws, 2)
	require.Equal(t, "low", ws[0].Level)
	require.Equal(t, 20.0, ws[0].Balance)
	require.Equal(t, "ok", ws[1].Level)
}

// ---------------------------------------------------------------- 令牌保护与运行

type fakeSiteReconEncryptor struct{}

func (fakeSiteReconEncryptor) Encrypt(p string) (string, error) { return "enc:" + reverse(p), nil }
func (fakeSiteReconEncryptor) Decrypt(c string) (string, error) {
	if !strings.HasPrefix(c, "enc:") {
		return "", errors.New("bad ciphertext")
	}
	return reverse(strings.TrimPrefix(c, "enc:")), nil
}

func reverse(s string) string {
	r := []rune(s)
	for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {
		r[i], r[j] = r[j], r[i]
	}
	return string(r)
}

func newTestSiteReconService(t *testing.T) (*SiteReconService, *upstreamBillingProbeSettingRepo) {
	settings := &upstreamBillingProbeSettingRepo{}
	s := newSiteReconService(newSiteReconTestDB(t), false, nil, settings, fakeSiteReconEncryptor{}, nil)
	t.Cleanup(s.Stop)
	return s, settings
}

func TestSiteReconTokensAreEncryptedAndNeverReturned(t *testing.T) {
	s, _ := newTestSiteReconService(t)
	ctx := context.Background()
	require.NoError(t, s.SetToken(ctx, "https://Up.Example.com/console", "access-token-ABCD", 7))
	var cipher string
	require.NoError(t, s.store.db.QueryRow(`SELECT token_ciphertext FROM site_recon_upstream_tokens WHERE host = ?`, "up.example.com").Scan(&cipher))
	require.NotContains(t, cipher, "access-token-ABCD")

	view, err := s.ListTokens(ctx)
	require.NoError(t, err)
	require.Len(t, view.Tokens, 1)
	require.Equal(t, "****ABCD", view.Tokens[0].Hint)
	require.True(t, view.Tokens[0].Readable)
	require.NotContains(t, fmt.Sprintf("%+v", view), "access-token")

	got := s.decryptTokens(ctx, func(string) {})
	require.Equal(t, map[string]string{"up.example.com": "access-token-ABCD"}, got)

	require.ErrorIs(t, s.SetToken(ctx, "bad host!", "x", 1), ErrSiteReconInvalidInput)
	require.ErrorIs(t, s.SetToken(ctx, "up.example.com", "  ", 1), ErrSiteReconInvalidInput)
	require.NoError(t, s.DeleteToken(ctx, "up.example.com"))
	require.ErrorIs(t, s.DeleteToken(ctx, "up.example.com"), ErrSiteReconNotFound)
}

func TestSiteReconBarkURLStoredEncrypted(t *testing.T) {
	s, settings := newTestSiteReconService(t)
	ctx := context.Background()
	bark := "https://api.day.app/DEVICEKEY123"
	view, err := s.UpdateConfig(ctx, SiteReconConfigUpdate{SiteReconConfig: defaultSiteReconConfig(), BarkURL: &bark})
	require.NoError(t, err)
	require.True(t, view.BarkConfigured)
	require.Equal(t, "https://api.day.app/****", view.BarkHint)
	raw, _ := settings.GetValue(ctx, siteReconBarkURLKey)
	require.NotContains(t, raw, "DEVICEKEY123")
	require.Equal(t, bark, s.barkURL(ctx))

	bad := "ftp://x"
	_, err = s.UpdateConfig(ctx, SiteReconConfigUpdate{SiteReconConfig: defaultSiteReconConfig(), BarkURL: &bad})
	require.Error(t, err)
}

func TestSiteReconServiceManualRunEndToEnd(t *testing.T) {
	s, _ := newTestSiteReconService(t)
	site := &fakeSiteReconSite{world: map[int64]*fakeSiteReconAccount{
		1: {name: "A", key: "ka", usage: map[string][2]float64{srD3: {10, 100}}},
	}}
	s.newSite = func(string, *time.Location) siteReconSite { return site }
	s.openUpstream = func(context.Context, *siteReconUnit) (siteReconUpstream, string, string) {
		return &fakeSiteReconUpstream{costs: map[string][2]float64{srD3: {10, 40}}}, SiteReconStatusOK, ""
	}
	var pushed atomic.Int32
	s.sendBark = func(_ context.Context, _ string, fields map[string]string) error {
		pushed.Add(1)
		require.Contains(t, fields["title"], "毛利 ¥60.00")
		return nil
	}
	bark := "https://api.day.app/KEY"
	_, err := s.UpdateConfig(context.Background(), SiteReconConfigUpdate{SiteReconConfig: defaultSiteReconConfig(), BarkURL: &bark})
	require.NoError(t, err)
	s.now = func() time.Time { return time.Date(2026, 11, 10, 9, 0, 0, 0, time.FixedZone("BJ", 8*3600)) }

	id, err := s.StartManualRun(context.Background(), SiteReconStartInput{Date: srD3, Days: 1, Push: true})
	require.NoError(t, err)
	require.Eventually(t, func() bool {
		r, err := s.GetRun(context.Background(), id)
		return err == nil && r.Status != SiteReconRunRunning
	}, 5*time.Second, 20*time.Millisecond)
	r, err := s.GetRun(context.Background(), id)
	require.NoError(t, err)
	require.Equal(t, SiteReconRunOK, r.Status)
	require.Contains(t, r.Log, "已推送到手机")
	require.Equal(t, int32(1), pushed.Load())

	ov, err := s.Overview(context.Background(), "")
	require.NoError(t, err)
	require.Equal(t, srD3, ov.Day)
	require.Equal(t, 100.0, ov.Totals.Rev)
	require.Equal(t, 40.0, ov.Totals.Cost)

	_, err = s.StartManualRun(context.Background(), SiteReconStartInput{Date: "2026-12-01"})
	require.Error(t, err) // 未来日期
}

// 凭据加密器尚未初始化持久密钥：首次保存令牌时自动初始化，之后可正常解密。
type lazySiteReconEncryptor struct {
	fakeSiteReconEncryptor
	ready bool
	inits int
}

func (e *lazySiteReconEncryptor) Encrypt(p string) (string, error) {
	if !e.ready {
		return "", errors.New("credential encryption is unavailable")
	}
	return e.fakeSiteReconEncryptor.Encrypt(p)
}

func (e *lazySiteReconEncryptor) EncryptionStatus() (CredentialEncryptionStatus, error) {
	return CredentialEncryptionStatus{Configured: e.ready}, nil
}

func (e *lazySiteReconEncryptor) InitializeEncryption() (CredentialEncryptionStatus, error) {
	e.inits++
	e.ready = true
	return CredentialEncryptionStatus{Configured: true}, nil
}

func TestSiteReconTokenInitializesPersistentKeyOnFirstSave(t *testing.T) {
	enc := &lazySiteReconEncryptor{}
	s := newSiteReconService(newSiteReconTestDB(t), false, nil, &upstreamBillingProbeSettingRepo{}, enc, nil)
	t.Cleanup(s.Stop)
	require.NoError(t, s.SetToken(context.Background(), "up.example.com", "access-token-ABCD", 1))
	require.Equal(t, 1, enc.inits)
	require.Equal(t, map[string]string{"up.example.com": "access-token-ABCD"}, s.decryptTokens(context.Background(), func(string) {}))
}
