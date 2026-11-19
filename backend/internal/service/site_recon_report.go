package service

import (
	"context"
	"database/sql"
	"fmt"
	"math"
	"sort"
	"strings"
)

// 站点对账报表：从自有表算出页面和推送要用的数（只读）。

// SiteReconTotals 某天汇总；毛利只算拿到上游成本的账号，否则缺成本的账号会把利润抬虚。
type SiteReconTotals struct {
	Rev     float64 `json:"rev"`
	Cost    float64 `json:"cost"`
	Missing float64 `json:"missing"`
	All     float64 `json:"all"`
	Has     bool    `json:"has"`
	Scanned int64   `json:"scanned"`
}

type SiteReconPnLAccount struct {
	ID       int64     `json:"id"`
	Name     string    `json:"name"`
	Upstream string    `json:"upstream"`
	Rev      float64   `json:"rev"`
	Cost     *float64  `json:"cost"`
	Profit   *float64  `json:"profit"`
	Req      int64     `json:"req"`
	Rates    []float64 `json:"rates"`
	missing  bool
	cost     float64
}

type SiteReconGroupPnL struct {
	Name     string                 `json:"name"`
	Rev      float64                `json:"rev"`
	Cost     *float64               `json:"cost"`
	Missing  float64                `json:"missing"`
	Known    float64                `json:"known"`
	Profit   *float64               `json:"profit"`
	Margin   *float64               `json:"margin"`
	Req      int64                  `json:"req"`
	Rates    []float64              `json:"rates"`
	Accounts []*SiteReconPnLAccount `json:"accounts"`
	cost     float64
}

type SiteReconTrendPoint struct {
	Day     string  `json:"day"`
	Rev     float64 `json:"rev"`
	Cost    float64 `json:"cost"`
	Missing float64 `json:"missing"`
}

type SiteReconHistoryRow struct {
	Day        string   `json:"day"`
	Busy       int64    `json:"busy"`
	Done       int64    `json:"done"`
	Rev        *float64 `json:"rev"`
	Cost       *float64 `json:"cost"`
	Our        *int64   `json:"our_requests"`
	Up         *int64   `json:"up_requests"`
	RevMissing *float64 `json:"rev_missing"`
	Scanned    int64    `json:"scanned"`
}

type SiteReconAccountRow struct {
	SiteReconDaily
	Profit *float64 `json:"profit"`
	Leak   *float64 `json:"leak"`
	Idle   bool     `json:"idle"`
}

type SiteReconWallet struct {
	Host     string   `json:"host"`
	Balance  float64  `json:"balance"`
	Burn     float64  `json:"burn"`
	DaysLeft *float64 `json:"days_left"`
	Level    string   `json:"level"` // empty / low / ok / idle
	Names    []string `json:"names"`
	top      float64
}

type SiteReconPending struct {
	AccountID   int64   `json:"account_id"`
	AccountName string  `json:"account_name"`
	Upstream    string  `json:"upstream"`
	Error       string  `json:"error"`
	Revenue     float64 `json:"revenue"`
	Rev7        float64 `json:"rev7"`
}

type SiteReconModelReport struct {
	AccountID   int64  `json:"account_id"`
	AccountName string `json:"account_name"`
	siteReconModelRow
}

type SiteReconAnomalyStat struct {
	AccountID   int64   `json:"account_id"`
	AccountName string  `json:"account_name"`
	Kind        string  `json:"kind"`
	Model       string  `json:"model"`
	Detail      string  `json:"detail"`
	Requests    int64   `json:"requests"`
	ListCost    float64 `json:"list_cost"`
	ActualCost  float64 `json:"actual_cost"`
}

type SiteReconAnomalySample struct {
	AccountID  int64   `json:"account_id"`
	LogID      int64   `json:"log_id"`
	Kind       string  `json:"kind"`
	CreatedAt  string  `json:"created_at"`
	UserID     int64   `json:"user_id"`
	Model      string  `json:"model"`
	TotalCost  float64 `json:"total_cost"`
	ActualCost float64 `json:"actual_cost"`
	Tokens     int64   `json:"tokens"`
}

type SiteReconAnomalyCounts struct {
	Day      string  `json:"day,omitempty"`
	High     int64   `json:"high"`
	Mid      int64   `json:"mid"`
	Low      int64   `json:"low"`
	HighList float64 `json:"high_list"`
}

type SiteReconAlert struct {
	Level string `json:"level"` // bad / warn / info
	Text  string `json:"text"`
}

func siteReconKindLevel(kind string) string {
	switch kind {
	case SiteReconKindZeroPaid, SiteReconKindUnpriced:
		return "high"
	case SiteReconKindMismatch:
		return "low"
	default:
		return "mid"
	}
}

func (s *siteReconStore) daysAvailable(ctx context.Context) ([]string, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT DISTINCT day FROM site_recon_daily ORDER BY day DESC`)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []string
	for rows.Next() {
		var d string
		if err := rows.Scan(&d); err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

func (s *siteReconStore) dayTotals(ctx context.Context, day string) (SiteReconTotals, error) {
	var t SiteReconTotals
	var rev, cost, missing, all sql.NullFloat64
	var scanned sql.NullInt64
	var n int64
	err := s.db.QueryRowContext(ctx, s.q(`SELECT SUM(CASE WHEN cost IS NOT NULL THEN revenue END), SUM(cost),
SUM(CASE WHEN cost IS NULL THEN revenue END), SUM(revenue),
SUM(CASE WHEN scanned_rows IS NOT NULL THEN 1 ELSE 0 END), COUNT(*) FROM site_recon_daily WHERE day = ?`), day).
		Scan(&rev, &cost, &missing, &all, &scanned, &n)
	if err != nil {
		return t, err
	}
	t.Rev, t.Cost, t.Missing, t.All = rev.Float64, cost.Float64, missing.Float64, all.Float64
	t.Has = n > 0
	t.Scanned = scanned.Int64
	return t, nil
}

func addRate(rates []float64, r float64) []float64 {
	for _, x := range rates {
		if x == r {
			return rates
		}
	}
	return append(rates, r)
}

// groupPnL 分组盈亏。分组成本在对账时已按“先按模型、再按本站原价占比”分摊好。
func (s *siteReconStore) groupPnL(ctx context.Context, day string) ([]*SiteReconGroupPnL, error) {
	rows, err := s.db.QueryContext(ctx, s.q(`SELECT g.group_name, g.rate_multiplier, g.requests, g.revenue, g.cost,
g.account_id, d.account_name, d.upstream FROM site_recon_groups g
JOIN site_recon_daily d ON d.day = g.day AND d.account_id = g.account_id WHERE g.day = ?`), day)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	groups := map[string]*SiteReconGroupPnL{}
	accts := map[string]map[int64]*SiteReconPnLAccount{}
	for rows.Next() {
		var name, acctName, upstream string
		var rate, rev float64
		var req, acctID int64
		var cost sql.NullFloat64
		if err := rows.Scan(&name, &rate, &req, &rev, &cost, &acctID, &acctName, &upstream); err != nil {
			return nil, err
		}
		g := groups[name]
		if g == nil {
			g = &SiteReconGroupPnL{Name: name}
			groups[name] = g
			accts[name] = map[int64]*SiteReconPnLAccount{}
		}
		a := accts[name][acctID]
		if a == nil {
			a = &SiteReconPnLAccount{ID: acctID, Name: acctName, Upstream: upstream}
			accts[name][acctID] = a
		}
		g.Rev += rev
		a.Rev += rev
		g.Req += req
		a.Req += req
		g.Rates = addRate(g.Rates, rate)
		a.Rates = addRate(a.Rates, rate)
		if !cost.Valid {
			g.Missing += rev
			a.missing = true
		} else {
			g.cost += cost.Float64
			a.cost += cost.Float64
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	out := make([]*SiteReconGroupPnL, 0, len(groups))
	for name, g := range groups {
		g.Known = g.Rev - g.Missing
		if g.Known != 0 || g.cost != 0 {
			p := g.Known - g.cost
			g.Profit = &p
			g.Cost = siteReconF64(g.cost)
			if g.Known > 0 {
				g.Margin = siteReconF64(p / g.Known)
			}
		}
		sort.Float64s(g.Rates)
		for _, a := range accts[name] {
			sort.Float64s(a.Rates)
			if !a.missing {
				a.Cost = siteReconF64(a.cost)
				a.Profit = siteReconF64(a.Rev - a.cost)
			}
			g.Accounts = append(g.Accounts, a)
		}
		sort.Slice(g.Accounts, func(i, j int) bool { return g.Accounts[i].Rev > g.Accounts[j].Rev })
		out = append(out, g)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Rev != out[j].Rev {
			return out[i].Rev > out[j].Rev
		}
		return out[i].Name < out[j].Name
	})
	return out, nil
}

// trend 截至 day 的近 n 天，旧的在前。
func (s *siteReconStore) trend(ctx context.Context, day string, n int) ([]SiteReconTrendPoint, error) {
	rows, err := s.db.QueryContext(ctx, s.q(`SELECT day, SUM(CASE WHEN cost IS NOT NULL THEN revenue END), SUM(cost),
SUM(CASE WHEN cost IS NULL THEN revenue END) FROM site_recon_daily WHERE day <= ? AND day > ? GROUP BY day ORDER BY day`),
		day, siteReconShiftDay(day, -n))
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []SiteReconTrendPoint
	for rows.Next() {
		var p SiteReconTrendPoint
		var rev, cost, missing sql.NullFloat64
		if err := rows.Scan(&p.Day, &rev, &cost, &missing); err != nil {
			return nil, err
		}
		p.Rev, p.Cost, p.Missing = rev.Float64, cost.Float64, missing.Float64
		out = append(out, p)
	}
	return out, rows.Err()
}

// history 每日历史。“有流量”= 两边任一边有请求；“已对账”= 其中拿到上游成本的。
func (s *siteReconStore) history(ctx context.Context, n int) ([]SiteReconHistoryRow, error) {
	rows, err := s.db.QueryContext(ctx, s.q(`SELECT day,
SUM(CASE WHEN COALESCE(our_requests,0) > 0 OR COALESCE(up_requests,0) > 0 THEN 1 ELSE 0 END),
SUM(CASE WHEN (COALESCE(our_requests,0) > 0 OR COALESCE(up_requests,0) > 0) AND cost IS NOT NULL THEN 1 ELSE 0 END),
SUM(CASE WHEN cost IS NOT NULL THEN revenue END), SUM(cost),
SUM(CASE WHEN cost IS NOT NULL THEN our_requests END), SUM(CASE WHEN cost IS NOT NULL THEN up_requests END),
SUM(CASE WHEN cost IS NULL THEN revenue END), SUM(CASE WHEN scanned_rows IS NOT NULL THEN 1 ELSE 0 END)
FROM site_recon_daily GROUP BY day ORDER BY day DESC LIMIT ?`), n)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []SiteReconHistoryRow
	for rows.Next() {
		var r SiteReconHistoryRow
		var rev, cost, revMissing sql.NullFloat64
		var our, up sql.NullInt64
		if err := rows.Scan(&r.Day, &r.Busy, &r.Done, &rev, &cost, &our, &up, &revMissing, &r.Scanned); err != nil {
			return nil, err
		}
		if rev.Valid {
			r.Rev = siteReconF64(rev.Float64)
		}
		if cost.Valid {
			r.Cost = siteReconF64(cost.Float64)
		}
		if our.Valid {
			r.Our = siteReconI64(our.Int64)
		}
		if up.Valid {
			r.Up = siteReconI64(up.Int64)
		}
		if revMissing.Valid {
			r.RevMissing = siteReconF64(revMissing.Float64)
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// siteReconLeak 上游比本站多出来的请求，按这个账号当天的平均客单价估算少收了多少。只是量级参考。
func siteReconLeak(revenue float64, our int64, up *int64) *float64 {
	if our == 0 || up == nil || *up <= our {
		return nil
	}
	return siteReconF64(float64(*up-our) * revenue / float64(our))
}

func (s *siteReconStore) accounts(ctx context.Context, day string) ([]SiteReconAccountRow, error) {
	rows, err := s.db.QueryContext(ctx, s.q(`SELECT `+siteReconDailyColumns+` FROM site_recon_daily WHERE day = ? ORDER BY revenue DESC, account_id`), day)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []SiteReconAccountRow
	for rows.Next() {
		d, err := scanSiteReconDaily(rows)
		if err != nil {
			return nil, err
		}
		a := SiteReconAccountRow{SiteReconDaily: *d}
		if d.Cost != nil {
			a.Profit = siteReconF64(d.Revenue - *d.Cost)
		}
		a.Leak = siteReconLeak(d.Revenue, d.OurRequests, d.UpRequests)
		a.Idle = d.OurRequests == 0 && (d.UpRequests == nil || *d.UpRequests == 0)
		out = append(out, a)
	}
	return out, rows.Err()
}

type siteReconGap struct {
	Requests int64
	Leak     float64
	Accounts int
}

func siteReconRequestGap(accounts []SiteReconAccountRow) siteReconGap {
	var g siteReconGap
	for _, a := range accounts {
		if a.Leak == nil {
			continue
		}
		g.Requests += *a.UpRequests - a.OurRequests
		g.Leak += *a.Leak
		g.Accounts++
	}
	return g
}

// wallets 上游余额还能撑几天。同一上游里余额几乎相同的账号视为共用钱包，日耗合并计算。
// 日耗 = 近 3 天的日均上游扣费。余额是对账运行那一刻取的。
func (s *siteReconStore) wallets(ctx context.Context, day string, thresholdDays float64) ([]*SiteReconWallet, error) {
	rows, err := s.db.QueryContext(ctx, s.q(`SELECT d.account_id, d.account_name, d.upstream, d.up_balance,
(SELECT AVG(h.cost) FROM site_recon_daily h WHERE h.account_id = d.account_id AND h.cost IS NOT NULL
   AND h.day >= ? AND h.day <= d.day) AS burn
FROM site_recon_daily d WHERE d.day = ? AND d.up_balance IS NOT NULL ORDER BY d.upstream, d.up_balance`),
		siteReconShiftDay(day, -2), day)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []*SiteReconWallet
	for rows.Next() {
		var id int64
		var name, host string
		var bal float64
		var burn sql.NullFloat64
		if err := rows.Scan(&id, &name, &host, &bal, &burn); err != nil {
			return nil, err
		}
		var w *SiteReconWallet
		if n := len(out); n > 0 {
			w = out[n-1]
		}
		if w == nil || w.Host != host || math.Abs(bal-w.top) > math.Max(1.0, 0.01*math.Abs(bal)) {
			w = &SiteReconWallet{Host: host, Balance: bal, top: bal}
			out = append(out, w)
		}
		w.top = bal
		w.Burn += burn.Float64
		if name == "" {
			name = fmt.Sprintf("#%d", id)
		}
		w.Names = append(w.Names, name)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	for _, w := range out {
		if w.Burn > 0.01 {
			w.DaysLeft = siteReconF64(math.Max(w.Balance, 0) / w.Burn)
		}
		switch {
		case w.Burn <= 0.01:
			w.Level = "idle" // 近几天没花钱，余额多少都不急
		case w.Balance <= 0.01:
			w.Level = "empty"
		case thresholdDays > 0 && *w.DaysLeft < thresholdDays:
			w.Level = "low"
		default:
			w.Level = "ok"
		}
	}
	order := map[string]int{"empty": 0, "low": 1, "ok": 2, "idle": 3}
	sort.SliceStable(out, func(i, j int) bool {
		if order[out[i].Level] != order[out[j].Level] {
			return order[out[i].Level] < order[out[j].Level]
		}
		di, dj := 1e9, 1e9
		if out[i].DaysLeft != nil {
			di = *out[i].DaysLeft
		}
		if out[j].DaysLeft != nil {
			dj = *out[j].DaysLeft
		}
		return di < dj
	})
	return out, nil
}

// pending 拿不到上游成本的账号，按近 7 天收入排。
func (s *siteReconStore) pending(ctx context.Context, day string) ([]SiteReconPending, error) {
	rows, err := s.db.QueryContext(ctx, s.q(`SELECT d.account_id, d.account_name, d.upstream, d.error_text, d.revenue,
(SELECT COALESCE(SUM(h.revenue), 0) FROM site_recon_daily h WHERE h.account_id = d.account_id AND h.day >= ? AND h.day <= d.day) AS rev7
FROM site_recon_daily d WHERE d.day = ? AND d.error_text IS NOT NULL ORDER BY rev7 DESC, d.upstream`), siteReconShiftDay(day, -6), day)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []SiteReconPending
	for rows.Next() {
		var p SiteReconPending
		if err := rows.Scan(&p.AccountID, &p.AccountName, &p.Upstream, &p.Error, &p.Revenue, &p.Rev7); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

func (s *siteReconStore) modelReport(ctx context.Context, day string) ([]SiteReconModelReport, error) {
	rows, err := s.db.QueryContext(ctx, s.q(`SELECT m.account_id, COALESCE(d.account_name, ''), m.model, m.our_requests, m.up_requests,
m.revenue, m.cost, m.our_list_cost FROM site_recon_models m
LEFT JOIN site_recon_daily d ON d.day = m.day AND d.account_id = m.account_id WHERE m.day = ? ORDER BY m.revenue DESC, m.account_id, m.model`), day)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []SiteReconModelReport
	for rows.Next() {
		var r SiteReconModelReport
		var up sql.NullInt64
		var cost sql.NullFloat64
		if err := rows.Scan(&r.AccountID, &r.AccountName, &r.Model, &r.OurRequests, &up, &r.Revenue, &cost, &r.OurListCost); err != nil {
			return nil, err
		}
		if up.Valid {
			r.UpRequests = siteReconI64(up.Int64)
		}
		if cost.Valid {
			r.Cost = siteReconF64(cost.Float64)
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func (s *siteReconStore) anomalyStats(ctx context.Context, day string) ([]SiteReconAnomalyStat, error) {
	rows, err := s.db.QueryContext(ctx, s.q(`SELECT s.account_id, COALESCE(d.account_name, ''), s.kind, s.model, s.detail,
s.requests, s.list_cost, s.actual_cost FROM site_recon_anomaly_stats s
LEFT JOIN site_recon_daily d ON d.day = s.day AND d.account_id = s.account_id WHERE s.day = ? ORDER BY s.requests DESC, s.account_id`), day)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []SiteReconAnomalyStat
	for rows.Next() {
		var r SiteReconAnomalyStat
		if err := rows.Scan(&r.AccountID, &r.AccountName, &r.Kind, &r.Model, &r.Detail, &r.Requests, &r.ListCost, &r.ActualCost); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func (s *siteReconStore) anomalySamples(ctx context.Context, day string) ([]SiteReconAnomalySample, error) {
	rows, err := s.db.QueryContext(ctx, s.q(`SELECT account_id, log_id, kind, created_at, user_id, model, total_cost, actual_cost, tokens
FROM site_recon_anomalies WHERE day = ? ORDER BY created_at, log_id`), day)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []SiteReconAnomalySample
	for rows.Next() {
		var r SiteReconAnomalySample
		if err := rows.Scan(&r.AccountID, &r.LogID, &r.Kind, &r.CreatedAt, &r.UserID, &r.Model, &r.TotalCost, &r.ActualCost, &r.Tokens); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func (s *siteReconStore) anomalyCounts(ctx context.Context, day string) (SiteReconAnomalyCounts, error) {
	c := SiteReconAnomalyCounts{Day: day}
	rows, err := s.db.QueryContext(ctx, s.q(`SELECT kind, SUM(requests), SUM(list_cost) FROM site_recon_anomaly_stats WHERE day = ? GROUP BY kind`), day)
	if err != nil {
		return c, err
	}
	defer func() { _ = rows.Close() }()
	for rows.Next() {
		var kind string
		var n sql.NullInt64
		var lc sql.NullFloat64
		if err := rows.Scan(&kind, &n, &lc); err != nil {
			return c, err
		}
		switch siteReconKindLevel(kind) {
		case "high":
			c.High += n.Int64
			c.HighList += lc.Float64
		case "low":
			c.Low += n.Int64
		default:
			c.Mid += n.Int64
		}
	}
	return c, rows.Err()
}

func (s *siteReconStore) anomalyHistory(ctx context.Context, n int) ([]SiteReconAnomalyCounts, error) {
	rows, err := s.db.QueryContext(ctx, s.q(`SELECT DISTINCT day FROM site_recon_daily WHERE scanned_rows IS NOT NULL ORDER BY day DESC LIMIT ?`), n)
	if err != nil {
		return nil, err
	}
	var days []string
	for rows.Next() {
		var d string
		if err := rows.Scan(&d); err != nil {
			_ = rows.Close()
			return nil, err
		}
		days = append(days, d)
	}
	_ = rows.Close()
	out := make([]SiteReconAnomalyCounts, 0, len(days))
	for _, d := range days {
		c, err := s.anomalyCounts(ctx, d)
		if err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, nil
}

// alerts 当天需要人看一眼的事。level：bad（要处理）/ warn（留意）/ info。推送只带 bad 和 warn。
func (s *siteReconStore) alerts(ctx context.Context, day string, balanceDays float64) ([]SiteReconAlert, error) {
	var out []SiteReconAlert
	t, err := s.dayTotals(ctx, day)
	if err != nil {
		return nil, err
	}
	wallets, err := s.wallets(ctx, day, balanceDays)
	if err != nil {
		return nil, err
	}
	for _, w := range wallets {
		who := strings.Join(w.Names[:min(3, len(w.Names))], "、")
		if len(w.Names) > 3 {
			who += fmt.Sprintf(" 等 %d 个账号", len(w.Names))
		}
		switch w.Level {
		case "empty":
			out = append(out, SiteReconAlert{"bad", fmt.Sprintf("上游 %s 余额已用完（¥%s），近 3 天日均扣 ¥%s：%s",
				w.Host, siteReconFormatMoney(w.Balance), siteReconFormatMoney(w.Burn), who)})
		case "low":
			out = append(out, SiteReconAlert{"bad", fmt.Sprintf("上游 %s 余额 ¥%s，按日均 ¥%s 只够 %.1f 天：%s",
				w.Host, siteReconFormatMoney(w.Balance), siteReconFormatMoney(w.Burn), *w.DaysLeft, who)})
		}
	}
	c, err := s.anomalyCounts(ctx, day)
	if err != nil {
		return nil, err
	}
	if c.High > 0 {
		out = append(out, SiteReconAlert{"bad", fmt.Sprintf("%s 条请求漏扣或模型未定价（原价合计 ¥%s），见异常页",
			siteReconFormatInt(c.High), siteReconFormatMoney(c.HighList))})
	}
	todo, err := s.pending(ctx, day)
	if err != nil {
		return nil, err
	}
	if t.Missing > 0.005 {
		out = append(out, SiteReconAlert{"warn", fmt.Sprintf("¥%s 收入拿不到上游成本，未计入毛利（%d 个账号待处理，见明细页）",
			siteReconFormatMoney(t.Missing), len(todo))})
	}
	accts, err := s.accounts(ctx, day)
	if err != nil {
		return nil, err
	}
	gap := siteReconRequestGap(accts)
	if gap.Requests > 0 {
		level := "info"
		if gap.Leak >= math.Max(20.0, 0.01*t.Rev) {
			level = "warn"
		}
		out = append(out, SiteReconAlert{level, fmt.Sprintf("上游比本站多 %s 次请求（上游扣了钱、本站没记账），估算少收约 ¥%s",
			siteReconFormatInt(gap.Requests), siteReconFormatMoney(gap.Leak))})
	}
	if t.Has && t.Scanned == 0 {
		out = append(out, SiteReconAlert{"info", "这一天还没有逐条扫描，所以没有分组盈亏和异常数据；到“运行”页对这一天手动跑一次即可补上"})
	}
	return out, nil
}
