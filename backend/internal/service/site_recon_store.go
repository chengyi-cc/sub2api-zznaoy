package service

import (
	"context"
	"database/sql"
	"errors"
	"strconv"
	"strings"
	"sync"
	"time"
)

// siteReconStore 站点对账自有表的读写。SQL 只用 PostgreSQL / SQLite 通用语法（单测跑 SQLite），
// 占位符统一写 ?，PostgreSQL 下改写成 $n。日期与时间都按 TEXT 存（YYYY-MM-DD / RFC3339），
// 与原 Python 服务一致，也避免两种数据库的时间类型差异。
type siteReconStore struct {
	db       *sql.DB
	postgres bool
	mu       sync.Mutex // 串行化“先读后写”的合并更新
}

func newSiteReconStore(db *sql.DB, postgres bool) *siteReconStore {
	return &siteReconStore{db: db, postgres: postgres}
}

func (s *siteReconStore) q(query string) string {
	if !s.postgres {
		return query
	}
	var b strings.Builder
	n := 0
	for _, c := range query {
		if c == '?' {
			n++
			b.WriteString("$" + strconv.Itoa(n))
			continue
		}
		b.WriteRune(c)
	}
	return b.String()
}

func siteReconPlaceholders(n int) string {
	if n <= 0 {
		return "NULL"
	}
	return strings.TrimSuffix(strings.Repeat("?,", n), ",")
}

func siteReconInt64Args(ids []int64) []any {
	out := make([]any, len(ids))
	for i, v := range ids {
		out[i] = v
	}
	return out
}

// siteReconSQLiteDDL 单测用的建表语句（生产环境由 migrations/253_site_recon.sql 建表，结构保持一致）。
const siteReconSQLiteDDL = `
CREATE TABLE IF NOT EXISTS site_recon_daily (
  day TEXT NOT NULL, account_id BIGINT NOT NULL, account_name TEXT NOT NULL DEFAULT '',
  upstream TEXT NOT NULL DEFAULT '', unit_key TEXT NOT NULL DEFAULT '',
  our_requests BIGINT NOT NULL DEFAULT 0, up_requests BIGINT, our_list_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
  up_list_cost DOUBLE PRECISION, revenue DOUBLE PRECISION NOT NULL DEFAULT 0, cost DOUBLE PRECISION,
  up_balance DOUBLE PRECISION, scanned_rows BIGINT, anomalies BIGINT, mismatches BIGINT,
  error_text TEXT, updated_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (day, account_id));
CREATE TABLE IF NOT EXISTS site_recon_models (
  day TEXT NOT NULL, account_id BIGINT NOT NULL, model TEXT NOT NULL, our_requests BIGINT NOT NULL DEFAULT 0,
  up_requests BIGINT, revenue DOUBLE PRECISION NOT NULL DEFAULT 0, cost DOUBLE PRECISION,
  our_list_cost DOUBLE PRECISION NOT NULL DEFAULT 0, PRIMARY KEY (day, account_id, model));
CREATE TABLE IF NOT EXISTS site_recon_groups (
  day TEXT NOT NULL, account_id BIGINT NOT NULL, group_name TEXT NOT NULL, rate_multiplier DOUBLE PRECISION NOT NULL,
  requests BIGINT NOT NULL DEFAULT 0, list_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
  revenue DOUBLE PRECISION NOT NULL DEFAULT 0, cost DOUBLE PRECISION,
  PRIMARY KEY (day, account_id, group_name, rate_multiplier));
CREATE TABLE IF NOT EXISTS site_recon_anomalies (
  day TEXT NOT NULL, account_id BIGINT NOT NULL, log_id BIGINT NOT NULL, kind TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT '', user_id BIGINT NOT NULL DEFAULT 0, model TEXT NOT NULL DEFAULT '',
  total_cost DOUBLE PRECISION NOT NULL DEFAULT 0, actual_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
  tokens BIGINT NOT NULL DEFAULT 0, PRIMARY KEY (day, account_id, log_id));
CREATE TABLE IF NOT EXISTS site_recon_anomaly_stats (
  day TEXT NOT NULL, account_id BIGINT NOT NULL, kind TEXT NOT NULL, model TEXT NOT NULL, detail TEXT NOT NULL,
  requests BIGINT NOT NULL DEFAULT 0, list_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
  actual_cost DOUBLE PRECISION NOT NULL DEFAULT 0, PRIMARY KEY (day, account_id, kind, model, detail));
CREATE TABLE IF NOT EXISTS site_recon_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, trigger_kind TEXT NOT NULL, target TEXT NOT NULL DEFAULT '',
  days INT NOT NULL DEFAULT 0, started_at TEXT NOT NULL, finished_at TEXT NOT NULL DEFAULT '',
  heartbeat_at TEXT NOT NULL DEFAULT '', status TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '{}',
  log TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS site_recon_upstream_tokens (
  host TEXT PRIMARY KEY, token_ciphertext TEXT NOT NULL, token_hint TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL DEFAULT '', updated_by BIGINT NOT NULL DEFAULT 0);
`

// ---------------------------------------------------------------- daily

const siteReconDailyColumns = `day, account_id, account_name, upstream, unit_key, our_requests, up_requests,
our_list_cost, up_list_cost, revenue, cost, up_balance, scanned_rows, anomalies, mismatches, error_text, updated_at`

func scanSiteReconDaily(row interface{ Scan(...any) error }) (*SiteReconDaily, error) {
	var d SiteReconDaily
	var upReq, scanned, anomalies, mismatches sql.NullInt64
	var upList, cost, balance sql.NullFloat64
	var errText sql.NullString
	if err := row.Scan(&d.Day, &d.AccountID, &d.AccountName, &d.Upstream, &d.UnitKey, &d.OurRequests, &upReq,
		&d.OurListCost, &upList, &d.Revenue, &cost, &balance, &scanned, &anomalies, &mismatches, &errText, &d.UpdatedAt); err != nil {
		return nil, err
	}
	if upReq.Valid {
		d.UpRequests = siteReconI64(upReq.Int64)
	}
	if scanned.Valid {
		d.ScannedRows = siteReconI64(scanned.Int64)
	}
	if anomalies.Valid {
		d.Anomalies = siteReconI64(anomalies.Int64)
	}
	if mismatches.Valid {
		d.Mismatches = siteReconI64(mismatches.Int64)
	}
	if upList.Valid {
		d.UpListCost = siteReconF64(upList.Float64)
	}
	if cost.Valid {
		d.Cost = siteReconF64(cost.Float64)
	}
	if balance.Valid {
		d.UpBalance = siteReconF64(balance.Float64)
	}
	if errText.Valid {
		d.Error = siteReconStr(errText.String)
	}
	return &d, nil
}

func (s *siteReconStore) getDaily(ctx context.Context, day string, accountID int64) (*SiteReconDaily, error) {
	row := s.db.QueryRowContext(ctx, s.q(`SELECT `+siteReconDailyColumns+` FROM site_recon_daily WHERE day = ? AND account_id = ?`), day, accountID)
	d, err := scanSiteReconDaily(row)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	return d, err
}

func nullInt64(p *int64) any {
	if p == nil {
		return nil
	}
	return *p
}

func nullFloat(p *float64) any {
	if p == nil {
		return nil
	}
	return *p
}

func nullString(p *string) any {
	if p == nil {
		return nil
	}
	return *p
}

func (s *siteReconStore) upsertDaily(ctx context.Context, d *SiteReconDaily) error {
	_, err := s.db.ExecContext(ctx, s.q(`INSERT INTO site_recon_daily (`+siteReconDailyColumns+`)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT (day, account_id) DO UPDATE SET account_name = excluded.account_name, upstream = excluded.upstream,
unit_key = excluded.unit_key, our_requests = excluded.our_requests, up_requests = excluded.up_requests,
our_list_cost = excluded.our_list_cost, up_list_cost = excluded.up_list_cost, revenue = excluded.revenue,
cost = excluded.cost, up_balance = excluded.up_balance, scanned_rows = excluded.scanned_rows,
anomalies = excluded.anomalies, mismatches = excluded.mismatches, error_text = excluded.error_text,
updated_at = excluded.updated_at`),
		d.Day, d.AccountID, d.AccountName, d.Upstream, d.UnitKey, d.OurRequests, nullInt64(d.UpRequests),
		d.OurListCost, nullFloat(d.UpListCost), d.Revenue, nullFloat(d.Cost), nullFloat(d.UpBalance),
		nullInt64(d.ScannedRows), nullInt64(d.Anomalies), nullInt64(d.Mismatches), nullString(d.Error), d.UpdatedAt)
	return err
}

// mergeDaily 部分字段更新：已有行保留 mutate 未改动的字段（如历史扫描结果）。
func (s *siteReconStore) mergeDaily(ctx context.Context, day string, accountID int64, now time.Time, mutate func(*SiteReconDaily)) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	d, err := s.getDaily(ctx, day, accountID)
	if err != nil {
		return err
	}
	if d == nil {
		d = &SiteReconDaily{Day: day, AccountID: accountID}
	}
	mutate(d)
	d.UpdatedAt = now.Format(time.RFC3339)
	return s.upsertDaily(ctx, d)
}

func (s *siteReconStore) accountsWithRows(ctx context.Context, first, last string, ids []int64) (map[int64]bool, error) {
	out := make(map[int64]bool)
	query := `SELECT DISTINCT account_id FROM site_recon_daily WHERE day >= ? AND day <= ?`
	args := []any{first, last}
	if ids != nil {
		if len(ids) == 0 {
			return out, nil
		}
		query += ` AND account_id IN (` + siteReconPlaceholders(len(ids)) + `)`
		args = append(args, siteReconInt64Args(ids)...)
	}
	rows, err := s.db.QueryContext(ctx, s.q(query), args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		out[id] = true
	}
	return out, rows.Err()
}

var siteReconPerDayTables = []string{"site_recon_models", "site_recon_groups", "site_recon_anomalies", "site_recon_anomaly_stats"}

// dropRows 删掉窗口内这些账号的行（daily 与各明细表）。
func (s *siteReconStore) dropRows(ctx context.Context, first, last string, ids []int64) error {
	if len(ids) == 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	args := append([]any{first, last}, siteReconInt64Args(ids)...)
	for _, table := range append([]string{"site_recon_daily"}, siteReconPerDayTables...) {
		if _, err := s.db.ExecContext(ctx, s.q(`DELETE FROM `+table+` WHERE day >= ? AND day <= ? AND account_id IN (`+siteReconPlaceholders(len(ids))+`)`), args...); err != nil {
			return err
		}
	}
	return nil
}

// otherAccountsForUnit 窗口内挂着这把 key（unit_key）、但不是 keep 这个账号的行。
func (s *siteReconStore) otherAccountsForUnit(ctx context.Context, first, last, unitKey string, keep int64) ([]int64, error) {
	rows, err := s.db.QueryContext(ctx, s.q(`SELECT DISTINCT account_id FROM site_recon_daily
WHERE day >= ? AND day <= ? AND unit_key = ? AND account_id <> ?`), first, last, unitKey, keep)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		out = append(out, id)
	}
	return out, rows.Err()
}

// hasUnallocatedGroups 这天是否有成本未分摊的分组行（扫描时拿不到上游成本）。
func (s *siteReconStore) hasUnallocatedGroups(ctx context.Context, day string, accountID int64) (bool, error) {
	var n int
	err := s.db.QueryRowContext(ctx, s.q(`SELECT COUNT(*) FROM site_recon_groups WHERE day = ? AND account_id = ? AND cost IS NULL`), day, accountID).Scan(&n)
	return n > 0, err
}

// siteReconDayDetail 一天逐条扫描后要整体替换的明细。
type siteReconDayDetail struct {
	Samples []siteReconSample
	Stats   map[siteReconStatKey]*siteReconStat
	Models  []siteReconModelRow
	Groups  []siteReconGroupRow
}

type siteReconModelRow struct {
	Model       string   `json:"model"`
	OurRequests int64    `json:"our_requests"`
	UpRequests  *int64   `json:"up_requests"`
	Revenue     float64  `json:"revenue"`
	Cost        *float64 `json:"cost"`
	OurListCost float64  `json:"our_list_cost"`
}

type siteReconGroupRow struct {
	GroupName      string   `json:"group_name"`
	RateMultiplier float64  `json:"rate_multiplier"`
	Requests       int64    `json:"requests"`
	ListCost       float64  `json:"list_cost"`
	Revenue        float64  `json:"revenue"`
	Cost           *float64 `json:"cost"`
}

// replaceDayDetail 在一个事务里替换某天某账号的全部明细。
func (s *siteReconStore) replaceDayDetail(ctx context.Context, day string, accountID int64, d *siteReconDayDetail) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	for _, table := range siteReconPerDayTables {
		if _, err := tx.ExecContext(ctx, s.q(`DELETE FROM `+table+` WHERE day = ? AND account_id = ?`), day, accountID); err != nil {
			return err
		}
	}
	for _, b := range d.Samples {
		if _, err := tx.ExecContext(ctx, s.q(`INSERT INTO site_recon_anomalies (day, account_id, log_id, kind, created_at, user_id, model, total_cost, actual_cost, tokens)
VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT (day, account_id, log_id) DO NOTHING`),
			day, accountID, b.LogID, b.Kind, b.CreatedAt, b.UserID, b.Model, b.TotalCost, b.ActualCost, b.Tokens); err != nil {
			return err
		}
	}
	for k, v := range d.Stats {
		if _, err := tx.ExecContext(ctx, s.q(`INSERT INTO site_recon_anomaly_stats (day, account_id, kind, model, detail, requests, list_cost, actual_cost)
VALUES (?,?,?,?,?,?,?,?)`), day, accountID, k.Kind, k.Model, k.Detail, v.Requests, v.ListCost, v.ActualCost); err != nil {
			return err
		}
	}
	for _, m := range d.Models {
		if _, err := tx.ExecContext(ctx, s.q(`INSERT INTO site_recon_models (day, account_id, model, our_requests, up_requests, revenue, cost, our_list_cost)
VALUES (?,?,?,?,?,?,?,?)`), day, accountID, m.Model, m.OurRequests, nullInt64(m.UpRequests), m.Revenue, nullFloat(m.Cost), m.OurListCost); err != nil {
			return err
		}
	}
	for _, g := range d.Groups {
		if _, err := tx.ExecContext(ctx, s.q(`INSERT INTO site_recon_groups (day, account_id, group_name, rate_multiplier, requests, list_cost, revenue, cost)
VALUES (?,?,?,?,?,?,?,?)`), day, accountID, g.GroupName, g.RateMultiplier, g.Requests, g.ListCost, g.Revenue, nullFloat(g.Cost)); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// ---------------------------------------------------------------- runs

// SiteReconRun 一次对账运行记录。
type SiteReconRun struct {
	ID          int64          `json:"id"`
	Trigger     string         `json:"trigger"`
	Target      string         `json:"target"`
	Days        int            `json:"days"`
	StartedAt   string         `json:"started_at"`
	FinishedAt  string         `json:"finished_at"`
	HeartbeatAt string         `json:"-"`
	Status      string         `json:"status"`
	Summary     map[string]any `json:"summary"`
	Log         string         `json:"log,omitempty"`
}

const (
	SiteReconRunRunning = "running"
	SiteReconRunOK      = "ok"
	SiteReconRunFailed  = "failed"
	SiteReconRunAborted = "aborted"
)

func (s *siteReconStore) insertRun(ctx context.Context, trigger, target string, days int, now time.Time) (int64, error) {
	ts := now.Format(time.RFC3339)
	query := `INSERT INTO site_recon_runs (trigger_kind, target, days, started_at, heartbeat_at, status, summary, log)
VALUES (?,?,?,?,?,?,'{}','') RETURNING id`
	var id int64
	err := s.db.QueryRowContext(ctx, s.q(query), trigger, target, days, ts, ts, SiteReconRunRunning).Scan(&id)
	return id, err
}

func (s *siteReconStore) heartbeatRun(ctx context.Context, id int64, log string, now time.Time) error {
	_, err := s.db.ExecContext(ctx, s.q(`UPDATE site_recon_runs SET heartbeat_at = ?, log = ? WHERE id = ? AND status = ?`),
		now.Format(time.RFC3339), log, id, SiteReconRunRunning)
	return err
}

func (s *siteReconStore) finishRun(ctx context.Context, id int64, target string, days int, status, summary, log string, now time.Time) error {
	_, err := s.db.ExecContext(ctx, s.q(`UPDATE site_recon_runs SET target = ?, days = ?, finished_at = ?, heartbeat_at = ?, status = ?, summary = ?, log = ? WHERE id = ?`),
		target, days, now.Format(time.RFC3339), now.Format(time.RFC3339), status, summary, log, id)
	return err
}

// abortStaleRuns 心跳过期的“进行中”记录：进程已经不在了，标成中断。
func (s *siteReconStore) abortStaleRuns(ctx context.Context, staleBefore time.Time) error {
	_, err := s.db.ExecContext(ctx, s.q(`UPDATE site_recon_runs SET status = ? WHERE status = ? AND heartbeat_at < ?`),
		SiteReconRunAborted, SiteReconRunRunning, staleBefore.Format(time.RFC3339))
	return err
}

// activeRun 心跳仍新鲜的进行中记录（任何实例都算）。
func (s *siteReconStore) activeRun(ctx context.Context, staleBefore time.Time) (*SiteReconRun, error) {
	runs, err := s.queryRuns(ctx, `WHERE status = ? AND heartbeat_at >= ? ORDER BY id DESC LIMIT 1`, true, SiteReconRunRunning, staleBefore.Format(time.RFC3339))
	if err != nil || len(runs) == 0 {
		return nil, err
	}
	return &runs[0], nil
}

func (s *siteReconStore) listRuns(ctx context.Context, limit int) ([]SiteReconRun, error) {
	return s.queryRuns(ctx, `ORDER BY id DESC LIMIT ?`, false, limit)
}

func (s *siteReconStore) getRun(ctx context.Context, id int64) (*SiteReconRun, error) {
	runs, err := s.queryRuns(ctx, `WHERE id = ?`, true, id)
	if err != nil || len(runs) == 0 {
		return nil, err
	}
	return &runs[0], nil
}

func (s *siteReconStore) lastOKSummary(ctx context.Context) (map[string]any, error) {
	runs, err := s.queryRuns(ctx, `WHERE status = ? ORDER BY id DESC LIMIT 1`, false, SiteReconRunOK)
	if err != nil || len(runs) == 0 {
		return nil, err
	}
	return runs[0].Summary, nil
}

func (s *siteReconStore) queryRuns(ctx context.Context, tail string, withLog bool, args ...any) ([]SiteReconRun, error) {
	logCol := "''"
	if withLog {
		logCol = "log"
	}
	rows, err := s.db.QueryContext(ctx, s.q(`SELECT id, trigger_kind, target, days, started_at, finished_at, heartbeat_at, status, summary, `+logCol+` FROM site_recon_runs `+tail), args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []SiteReconRun
	for rows.Next() {
		var r SiteReconRun
		var summary string
		if err := rows.Scan(&r.ID, &r.Trigger, &r.Target, &r.Days, &r.StartedAt, &r.FinishedAt, &r.HeartbeatAt, &r.Status, &summary, &r.Log); err != nil {
			return nil, err
		}
		r.Summary = siteReconDecodeSummary(summary)
		out = append(out, r)
	}
	return out, rows.Err()
}

// ---------------------------------------------------------------- tokens

type siteReconTokenRow struct {
	Host       string
	Ciphertext string
	Hint       string
	UpdatedAt  string
}

func (s *siteReconStore) listTokens(ctx context.Context) ([]siteReconTokenRow, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT host, token_ciphertext, token_hint, updated_at FROM site_recon_upstream_tokens ORDER BY host`)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []siteReconTokenRow
	for rows.Next() {
		var r siteReconTokenRow
		if err := rows.Scan(&r.Host, &r.Ciphertext, &r.Hint, &r.UpdatedAt); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func (s *siteReconStore) upsertToken(ctx context.Context, host, ciphertext, hint string, by int64, now time.Time) error {
	_, err := s.db.ExecContext(ctx, s.q(`INSERT INTO site_recon_upstream_tokens (host, token_ciphertext, token_hint, updated_at, updated_by)
VALUES (?,?,?,?,?) ON CONFLICT (host) DO UPDATE SET token_ciphertext = excluded.token_ciphertext,
token_hint = excluded.token_hint, updated_at = excluded.updated_at, updated_by = excluded.updated_by`),
		host, ciphertext, hint, now.Format(time.RFC3339), by)
	return err
}

func (s *siteReconStore) deleteToken(ctx context.Context, host string) (bool, error) {
	res, err := s.db.ExecContext(ctx, s.q(`DELETE FROM site_recon_upstream_tokens WHERE host = ?`), host)
	if err != nil {
		return false, err
	}
	n, _ := res.RowsAffected()
	return n > 0, nil
}
