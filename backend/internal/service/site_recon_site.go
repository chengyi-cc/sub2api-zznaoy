package service

import (
	"context"
	"database/sql"
	"net/url"
	"strings"
	"time"
)

// siteReconAccount 参与对账的本站 apikey 账号（凭证已由账号仓库解密，只在进程内使用）。
type siteReconAccount struct {
	ID       int64
	Name     string
	Status   string
	BaseURL  string
	APIKey   string
	ProxyURL string
}

// siteReconSite 本站数据源。生产实现直接查 usage_logs；单测用假的。
type siteReconSite interface {
	ListAPIKeyAccounts(ctx context.Context) ([]siteReconAccount, error)
	// WindowRequests 窗口 [first, last]（本地日期）内各账号的请求数。
	WindowRequests(ctx context.Context, first, last string) (map[int64]int64, error)
	// DaysStats 多个账号（共用同一上游 key）按天求和。
	DaysStats(ctx context.Context, ids []int64, days []string) (map[string]siteReconDayStats, error)
	// ScanDay 逐条扫描一天：异常、按分组×模型汇总。
	ScanDay(ctx context.Context, ids []int64, day string) (*siteReconScan, error)
}

const siteReconSiteQueryConcurrency = 4 // 打到生产库的对账查询并发上限

type siteReconPGSite struct {
	db       *sql.DB
	accounts AccountRepository
	tz       string
	loc      *time.Location
	sem      chan struct{}
}

func newSiteReconPGSite(db *sql.DB, accounts AccountRepository, tz string, loc *time.Location) *siteReconPGSite {
	return &siteReconPGSite{db: db, accounts: accounts, tz: tz, loc: loc, sem: make(chan struct{}, siteReconSiteQueryConcurrency)}
}

func (s *siteReconPGSite) acquire(ctx context.Context) (func(), error) {
	select {
	case s.sem <- struct{}{}:
		return func() { <-s.sem }, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// ListAPIKeyAccounts 所有 apikey 型账号，不论状态（停用 / 报错账号停用前产生的收入和成本是真实发生的）。
func (s *siteReconPGSite) ListAPIKeyAccounts(ctx context.Context) ([]siteReconAccount, error) {
	list, err := s.accounts.ListAllWithFilters(ctx, "", AccountTypeAPIKey, "", "", 0, "")
	if err != nil {
		return nil, err
	}
	out := make([]siteReconAccount, 0, len(list))
	for i := range list {
		a := &list[i]
		if a.Type != AccountTypeAPIKey {
			continue
		}
		item := siteReconAccount{
			ID:      a.ID,
			Name:    strings.TrimSpace(a.Name),
			Status:  a.Status,
			BaseURL: a.GetCredential("base_url"),
			APIKey:  a.GetCredential("api_key"),
		}
		if a.ProxyID != nil && a.Proxy != nil {
			item.ProxyURL = a.Proxy.URL()
		}
		out = append(out, item)
	}
	return out, nil
}

func (s *siteReconPGSite) window(first, last string) (time.Time, time.Time, error) {
	start, _, err := siteReconDayWindow(first, s.loc)
	if err != nil {
		return time.Time{}, time.Time{}, err
	}
	_, end, err := siteReconDayWindow(last, s.loc)
	return start, end, err
}

func (s *siteReconPGSite) WindowRequests(ctx context.Context, first, last string) (map[int64]int64, error) {
	start, end, err := s.window(first, last)
	if err != nil {
		return nil, err
	}
	release, err := s.acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer release()
	rows, err := s.db.QueryContext(ctx, `SELECT account_id, COUNT(*) FROM usage_logs
WHERE created_at >= $1 AND created_at < $2 GROUP BY account_id`, start, end)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	out := make(map[int64]int64)
	for rows.Next() {
		var id, n int64
		if err := rows.Scan(&id, &n); err != nil {
			return nil, err
		}
		out[id] = n
	}
	return out, rows.Err()
}

func siteReconPGIn(ids []int64, startIndex int) (string, []any) {
	var b strings.Builder
	args := make([]any, len(ids))
	for i, id := range ids {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString("$")
		b.WriteString(itoa(startIndex + i))
		args[i] = id
	}
	return b.String(), args
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

func (s *siteReconPGSite) DaysStats(ctx context.Context, ids []int64, days []string) (map[string]siteReconDayStats, error) {
	out := make(map[string]siteReconDayStats, len(days))
	for _, d := range days {
		out[d] = siteReconDayStats{}
	}
	if len(ids) == 0 || len(days) == 0 {
		return out, nil
	}
	first, last := days[0], days[0]
	for _, d := range days {
		if d < first {
			first = d
		}
		if d > last {
			last = d
		}
	}
	start, end, err := s.window(first, last)
	if err != nil {
		return nil, err
	}
	in, idArgs := siteReconPGIn(ids, 4)
	args := append([]any{s.tz, start, end}, idArgs...)
	release, err := s.acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer release()
	rows, err := s.db.QueryContext(ctx, `SELECT to_char(created_at AT TIME ZONE $1, 'YYYY-MM-DD') AS d, COUNT(*),
COALESCE(SUM(total_cost), 0), COALESCE(SUM(CASE WHEN billing_type = 0 THEN actual_cost ELSE 0 END), 0)
FROM usage_logs WHERE created_at >= $2 AND created_at < $3 AND account_id IN (`+in+`) GROUP BY 1`, args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	for rows.Next() {
		var d string
		var st siteReconDayStats
		if err := rows.Scan(&d, &st.Requests, &st.ListCost, &st.Revenue); err != nil {
			return nil, err
		}
		if _, ok := out[d]; ok {
			out[d] = st
		}
	}
	return out, rows.Err()
}

// siteReconScanBase 逐条扫描的公共 CTE：每条日志的异常种类在数据库里判定，顺序与原服务一致
// （漏扣 > 未定价 > 免费 > 模型不一致）。参数：$1 起、$2 止，$3.. 账号 ID。
const siteReconScanBase = `WITH base AS (
  SELECT ul.id, ul.created_at, ul.user_id, COALESCE(ul.model, '') AS model,
    COALESCE(NULLIF(ul.upstream_model, ''), ul.model, '') AS umodel,
    COALESCE(g.name, '#' || COALESCE(CAST(ul.group_id AS TEXT), '-')) AS gname,
    COALESCE(ul.rate_multiplier, 0) AS rate,
    ul.billing_type, COALESCE(ul.total_cost, 0) AS total_cost, COALESCE(ul.actual_cost, 0) AS actual_cost,
    COALESCE(ul.input_tokens, 0) + COALESCE(ul.output_tokens, 0) AS tokens,
    COALESCE(ul.upstream_response_model, '') AS resp_model,
    CASE
      WHEN ul.billing_type = 0 AND COALESCE(ul.total_cost, 0) > 0 AND COALESCE(ul.actual_cost, 0) = 0 THEN 'zero_paid'
      WHEN COALESCE(ul.input_tokens, 0) + COALESCE(ul.output_tokens, 0) > 0 AND COALESCE(ul.total_cost, 0) = 0 THEN 'unpriced'
      WHEN COALESCE(ul.rate_multiplier, 0) = 0 THEN 'free'
      WHEN ul.upstream_model_mismatch THEN 'mismatch'
      ELSE ''
    END AS kind
  FROM usage_logs ul LEFT JOIN groups g ON g.id = ul.group_id
  WHERE ul.created_at >= $1 AND ul.created_at < $2 AND ul.account_id IN (%IN%)
)`

func (s *siteReconPGSite) ScanDay(ctx context.Context, ids []int64, day string) (*siteReconScan, error) {
	res := &siteReconScan{
		Stats:       map[siteReconStatKey]*siteReconStat{},
		GroupModels: map[siteReconGroupModelKey]*siteReconAgg{},
	}
	if len(ids) == 0 {
		return res, nil
	}
	start, end, err := siteReconDayWindow(day, s.loc)
	if err != nil {
		return nil, err
	}
	in, idArgs := siteReconPGIn(ids, 3)
	base := strings.Replace(siteReconScanBase, "%IN%", in, 1)
	args := append([]any{start, end}, idArgs...)
	release, err := s.acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer release()

	rows, err := s.db.QueryContext(ctx, base+` SELECT gname, rate, umodel, COUNT(*), SUM(total_cost),
SUM(CASE WHEN billing_type = 0 THEN actual_cost ELSE 0 END) FROM base GROUP BY gname, rate, umodel`, args...)
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		var k siteReconGroupModelKey
		var a siteReconAgg
		if err := rows.Scan(&k.Group, &k.Rate, &k.Model, &a.Requests, &a.ListCost, &a.Revenue); err != nil {
			_ = rows.Close()
			return nil, err
		}
		res.GroupModels[k] = &a
		res.N += a.Requests
	}
	_ = rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}

	rows, err = s.db.QueryContext(ctx, base+` SELECT kind, model, CASE WHEN kind = 'mismatch' THEN resp_model ELSE '' END AS detail,
COUNT(*), SUM(total_cost), SUM(actual_cost) FROM base WHERE kind <> '' GROUP BY 1, 2, 3`, args...)
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		var k siteReconStatKey
		var st siteReconStat
		if err := rows.Scan(&k.Kind, &k.Model, &k.Detail, &st.Requests, &st.ListCost, &st.ActualCost); err != nil {
			_ = rows.Close()
			return nil, err
		}
		res.Stats[k] = &st
	}
	_ = rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}

	rows, err = s.db.QueryContext(ctx, base+` SELECT id, created_at, user_id, model, total_cost, actual_cost, tokens, kind FROM (
  SELECT base.*, ROW_NUMBER() OVER (PARTITION BY kind ORDER BY id) AS rn FROM base
  WHERE kind IN ('zero_paid', 'unpriced', 'free')
) t WHERE rn <= `+itoa(siteReconSampleCap)+` ORDER BY id`, args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	for rows.Next() {
		var smp siteReconSample
		var createdAt time.Time
		if err := rows.Scan(&smp.LogID, &createdAt, &smp.UserID, &smp.Model, &smp.TotalCost, &smp.ActualCost, &smp.Tokens, &smp.Kind); err != nil {
			return nil, err
		}
		smp.CreatedAt = createdAt.In(s.loc).Format(time.RFC3339)
		res.Samples = append(res.Samples, smp)
	}
	return res, rows.Err()
}

// siteReconOrigin base_url 的 scheme://host 与 host。
func siteReconOrigin(baseURL string) (origin, host string) {
	u, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil || u.Host == "" {
		return "", ""
	}
	return u.Scheme + "://" + u.Host, strings.ToLower(u.Host)
}
