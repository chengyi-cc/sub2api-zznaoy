package repository

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/lib/pq"
)

type candyMonitorRepository struct{ db *sql.DB }

func NewCandyMonitorRepository(db *sql.DB) service.CandyMonitorRepository {
	return &candyMonitorRepository{db: db}
}
func (r *candyMonitorRepository) Settings(ctx context.Context) (*service.CandyMonitorSettings, error) {
	s := &service.CandyMonitorSettings{}
	err := r.db.QueryRowContext(ctx, `SELECT enabled,model_id,interval_minutes,max_results FROM candy_monitor_settings WHERE singleton`).Scan(&s.Enabled, &s.ModelID, &s.IntervalMinutes, &s.MaxResults)
	return s, err
}
func (r *candyMonitorRepository) SaveSettings(ctx context.Context, s *service.CandyMonitorSettings) error {
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	var previousInterval int
	if err = tx.QueryRowContext(ctx, `SELECT interval_minutes FROM candy_monitor_settings WHERE singleton FOR UPDATE`).Scan(&previousInterval); err != nil {
		return err
	}
	if _, err = tx.ExecContext(ctx, `UPDATE candy_monitor_settings SET enabled=$1,model_id=$2,interval_minutes=$3,max_results=$4 WHERE singleton`, s.Enabled, s.ModelID, s.IntervalMinutes, s.MaxResults); err != nil {
		return err
	}
	if previousInterval != s.IntervalMinutes {
		if _, err = tx.ExecContext(ctx, `UPDATE candy_monitor_accounts SET next_run_at=NOW()+make_interval(mins=>$1),updated_at=NOW() WHERE use_defaults`, s.IntervalMinutes); err != nil {
			return err
		}
	}
	return tx.Commit()
}

const candyAccountFrom = ` FROM accounts a CROSS JOIN candy_monitor_settings s
 LEFT JOIN candy_monitor_accounts m ON m.account_id=a.id
 LEFT JOIN LATERAL (SELECT id,account_id,model_id,source,verdict,reason,actual,expected,duration_ms,started_at,finished_at,error_message FROM candy_monitor_results WHERE account_id=a.id ORDER BY id DESC LIMIT 1) latest ON TRUE `
const candyAccountHistory = ` LEFT JOIN LATERAL (SELECT COALESCE(jsonb_agg(to_jsonb(h) ORDER BY h.id DESC),'[]'::jsonb) AS history FROM (SELECT id,account_id,model_id,source,verdict,reason,actual,expected,duration_ms,started_at,finished_at FROM candy_monitor_results WHERE account_id=page.account_id AND verdict<>'running' ORDER BY id DESC LIMIT 10) h) recent ON TRUE `
const candyAccountColumns = `a.id AS account_id,a.name,a.platform,a.status,a.type,COALESCE(m.enabled,FALSE),COALESCE(m.use_defaults,TRUE),
 CASE WHEN COALESCE(m.use_defaults,TRUE) THEN s.model_id ELSE m.model_id END,
 CASE WHEN COALESCE(m.use_defaults,TRUE) THEN s.interval_minutes ELSE m.interval_minutes END,
 m.last_run_at,m.next_run_at,m.lease_until,COALESCE(to_jsonb(latest),'null'::jsonb),
 COALESCE(m.total_tests,0),COALESCE(m.answer_21_count,0),COALESCE(m.answer_29_count,0),COALESCE(m.other_answer_count,0),COALESCE(m.inconclusive_count,0)`
const candyEligible = `a.deleted_at IS NULL AND a.platform IN ('openai','anthropic','gemini') AND COALESCE(a.extra->>'synthetic_ui_test','false')<>'true' AND a.parent_account_id IS NULL`

func scanCandyAccounts(rows *sql.Rows) ([]service.CandyMonitorAccount, error) {
	out := make([]service.CandyMonitorAccount, 0)
	for rows.Next() {
		var a service.CandyMonitorAccount
		var raw []byte
		var historyRaw []byte
		if err := rows.Scan(&a.AccountID, &a.Name, &a.Platform, &a.Status, &a.Type, &a.Enabled, &a.UseDefaults, &a.ModelID, &a.IntervalMinutes, &a.LastRunAt, &a.NextRunAt, &a.RunningUntil, &raw, &a.TotalTests, &a.Answer21Count, &a.Answer29Count, &a.OtherAnswerCount, &a.InconclusiveCount, &historyRaw); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(raw, &a.Latest); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(historyRaw, &a.History); err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, rows.Err()
}
func (r *candyMonitorRepository) List(ctx context.Context, f service.CandyMonitorFilter) ([]service.CandyMonitorAccount, int64, error) {
	where := ` WHERE ` + candyEligible
	args := []any{}
	if f.Ungrouped {
		where += ` AND NOT EXISTS (SELECT 1 FROM account_groups ag WHERE ag.account_id=a.id)`
	} else if f.GroupID > 0 {
		args = append(args, f.GroupID)
		where += fmt.Sprintf(` AND EXISTS (SELECT 1 FROM account_groups ag WHERE ag.account_id=a.id AND ag.group_id=$%d)`, len(args))
	}
	if f.Search != "" {
		args = append(args, "%"+f.Search+"%")
		where += fmt.Sprintf(` AND (a.name ILIKE $%d OR a.id::text ILIKE $%d)`, len(args), len(args))
	}
	if f.Enabled != nil {
		args = append(args, *f.Enabled)
		where += fmt.Sprintf(` AND COALESCE(m.enabled,FALSE)=$%d`, len(args))
	} else if f.EnabledOnly {
		where += ` AND m.enabled`
	}
	if f.Platform != "" {
		args = append(args, f.Platform)
		where += fmt.Sprintf(` AND a.platform=$%d`, len(args))
	}
	if f.Type != "" {
		args = append(args, f.Type)
		where += fmt.Sprintf(` AND a.type=$%d`, len(args))
	}
	if f.PrivacyMode == "__unset__" {
		where += ` AND COALESCE(a.extra->>'privacy_mode','')=''`
	} else if f.PrivacyMode != "" {
		args = append(args, f.PrivacyMode)
		where += fmt.Sprintf(` AND a.extra->>'privacy_mode'=$%d`, len(args))
	}
	// Match the account management page's computed availability statuses.
	const notLimited = ` AND (a.rate_limit_reset_at IS NULL OR a.rate_limit_reset_at<=NOW())`
	const notTemporary = ` AND (a.temp_unschedulable_until IS NULL OR a.temp_unschedulable_until<=NOW())`
	switch f.Status {
	case "active":
		where += ` AND a.status='active' AND a.schedulable` + notLimited + notTemporary
	case "rate_limited":
		where += ` AND a.status='active' AND a.rate_limit_reset_at>NOW()` + notTemporary
	case "temp_unschedulable":
		where += ` AND a.status='active' AND a.temp_unschedulable_until>NOW()`
	case "unschedulable":
		where += ` AND a.status='active' AND NOT a.schedulable` + notLimited + notTemporary
	case "":
	default:
		args = append(args, f.Status)
		where += fmt.Sprintf(` AND a.status=$%d`, len(args))
	}
	if f.Verdict == "untested" {
		where += ` AND latest.id IS NULL`
	} else if f.Verdict != "" {
		args = append(args, f.Verdict)
		where += fmt.Sprintf(` AND latest.verdict=$%d`, len(args))
	}
	var total int64
	if err := r.db.QueryRowContext(ctx, `SELECT COUNT(*)`+candyAccountFrom+where, args...).Scan(&total); err != nil {
		return nil, 0, err
	}
	args = append(args, f.PageSize, (f.Page-1)*f.PageSize)
	// Load recent results only for the requested page, never for the whole fleet.
	rows, err := r.db.QueryContext(ctx, `SELECT page.*,recent.history FROM (SELECT `+candyAccountColumns+candyAccountFrom+where+fmt.Sprintf(` ORDER BY a.id DESC LIMIT $%d OFFSET $%d) page`, len(args)-1, len(args))+candyAccountHistory+` ORDER BY page.account_id DESC`, args...)
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = rows.Close() }()
	out, err := scanCandyAccounts(rows)
	return out, total, err
}
func (r *candyMonitorRepository) Configure(ctx context.Context, ids []int64, c service.CandyMonitorConfig) error {
	_, err := r.db.ExecContext(ctx, `INSERT INTO candy_monitor_accounts(account_id,enabled,use_defaults,model_id,interval_minutes,next_run_at)
 SELECT id,$2,$3,$4,$5,NOW()+make_interval(mins=>$5) FROM accounts WHERE id=ANY($1) AND deleted_at IS NULL
 ON CONFLICT(account_id) DO UPDATE SET enabled=EXCLUDED.enabled,use_defaults=EXCLUDED.use_defaults,model_id=EXCLUDED.model_id,
 interval_minutes=EXCLUDED.interval_minutes,next_run_at=EXCLUDED.next_run_at,updated_at=NOW()`, pq.Array(ids), c.Enabled, c.UseDefaults, c.ModelID, c.IntervalMinutes)
	return err
}
func (r *candyMonitorRepository) Due(ctx context.Context, limit int) ([]service.CandyMonitorAccount, error) {
	// Recover bounded leases after crashes; old workers cannot overwrite a newer run.
	if _, err := r.db.ExecContext(ctx, `WITH locked AS (
 SELECT m.account_id FROM candy_monitor_accounts m WHERE EXISTS (
 SELECT 1 FROM candy_monitor_results r WHERE r.account_id=m.account_id AND r.verdict='running' AND r.started_at<NOW()-INTERVAL '4 minutes') FOR UPDATE OF m SKIP LOCKED
 ), expired AS (
 UPDATE candy_monitor_results SET verdict='inconclusive',reason='interrupted',error_message='Test interrupted; retry required',finished_at=NOW()
 WHERE verdict='running' AND started_at<NOW()-INTERVAL '4 minutes' AND account_id IN (SELECT account_id FROM locked) RETURNING account_id
 ), counts AS (SELECT account_id,COUNT(*) AS n FROM expired GROUP BY account_id)
 UPDATE candy_monitor_accounts m SET total_tests=m.total_tests+c.n,inconclusive_count=m.inconclusive_count+c.n FROM counts c WHERE m.account_id=c.account_id`); err != nil {
		return nil, err
	}
	rows, err := r.db.QueryContext(ctx, `SELECT `+candyAccountColumns+`,'[]'::jsonb`+candyAccountFrom+` WHERE `+candyEligible+` AND s.enabled AND m.enabled AND m.next_run_at<=NOW() AND (m.lease_until IS NULL OR m.lease_until<=NOW()) ORDER BY m.next_run_at LIMIT $1`, limit)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	return scanCandyAccounts(rows)
}
func (r *candyMonitorRepository) SetEnabled(ctx context.Context, ids []int64, enabled bool) error {
	_, err := r.db.ExecContext(ctx, `UPDATE candy_monitor_accounts m SET enabled=$2,updated_at=NOW(),
 next_run_at=CASE WHEN $2 AND NOT m.enabled THEN NOW()+make_interval(mins=>CASE WHEN m.use_defaults THEN s.interval_minutes ELSE m.interval_minutes END) ELSE m.next_run_at END
 FROM candy_monitor_settings s WHERE m.account_id=ANY($1)`, pq.Array(ids), enabled)
	return err
}
func (r *candyMonitorRepository) AccountStates(ctx context.Context, ids []int64) ([]service.CandyMonitorAccountState, error) {
	rows, err := r.db.QueryContext(ctx, `SELECT page.*,recent.history FROM (SELECT a.id AS account_id,COALESCE(m.enabled,FALSE),COALESCE(m.use_defaults,TRUE),
 CASE WHEN COALESCE(m.use_defaults,TRUE) THEN s.model_id ELSE m.model_id END,
 CASE WHEN COALESCE(m.use_defaults,TRUE) THEN s.interval_minutes ELSE m.interval_minutes END,
 m.last_valid_answer,m.last_valid_at
 FROM accounts a CROSS JOIN candy_monitor_settings s LEFT JOIN candy_monitor_accounts m ON m.account_id=a.id
 WHERE `+candyEligible+` AND a.id=ANY($1)) page`+candyAccountHistory+` ORDER BY page.account_id`, pq.Array(ids))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := make([]service.CandyMonitorAccountState, 0)
	for rows.Next() {
		var item service.CandyMonitorAccountState
		var historyRaw []byte
		if err := rows.Scan(&item.AccountID, &item.Enabled, &item.UseDefaults, &item.ModelID, &item.IntervalMinutes, &item.LastValidAnswer, &item.LastValidAt, &historyRaw); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(historyRaw, &item.History); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func (r *candyMonitorRepository) SetMonitoring(ctx context.Context, id int64, enabled bool) error {
	// First-time opt-in inherits defaults. Re-enabling preserves custom settings
	// and makes the plan due; retries of an already enabled switch are idempotent.
	result, err := r.db.ExecContext(ctx, `INSERT INTO candy_monitor_accounts AS m(account_id,enabled,use_defaults,model_id,interval_minutes,next_run_at)
 SELECT a.id,$2,TRUE,s.model_id,s.interval_minutes,NOW() FROM accounts a CROSS JOIN candy_monitor_settings s WHERE a.id=$1 AND `+candyEligible+`
 ON CONFLICT(account_id) DO UPDATE SET enabled=EXCLUDED.enabled,
 next_run_at=CASE WHEN EXCLUDED.enabled AND NOT m.enabled THEN NOW() ELSE m.next_run_at END,updated_at=NOW()`, id, enabled)
	if err != nil {
		return err
	}
	n, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return sql.ErrNoRows
	}
	return nil
}
func (r *candyMonitorRepository) Begin(ctx context.Context, accountID int64, model string, scheduled bool) (*service.CandyMonitorResult, error) {
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tx.Rollback() }()
	if _, err = tx.ExecContext(ctx, `INSERT INTO candy_monitor_accounts(account_id) VALUES($1) ON CONFLICT DO NOTHING`, accountID); err != nil {
		return nil, err
	}
	claimed, err := tx.ExecContext(ctx, `UPDATE candy_monitor_accounts m SET lease_until=NOW()+INTERVAL '4 minutes',last_run_at=NOW(),
 next_run_at=NOW()+make_interval(mins=>CASE WHEN m.use_defaults THEN s.interval_minutes ELSE m.interval_minutes END)
 FROM candy_monitor_settings s WHERE m.account_id=$1 AND (m.lease_until IS NULL OR m.lease_until<=NOW())
 AND (NOT $2 OR (s.enabled AND m.enabled AND m.next_run_at<=NOW()))`, accountID, scheduled)
	if err != nil {
		return nil, err
	}
	n, err := claimed.RowsAffected()
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, service.ErrCandyMonitorBusy
	}
	// Re-read the effective model while holding the account lock. A template or
	// account edit between the due scan and this claim must not use the old model.
	if scheduled {
		if err = tx.QueryRowContext(ctx, `SELECT CASE WHEN m.use_defaults THEN s.model_id ELSE m.model_id END FROM candy_monitor_accounts m CROSS JOIN candy_monitor_settings s WHERE m.account_id=$1`, accountID).Scan(&model); err != nil {
			return nil, err
		}
	}
	source := "manual"
	if scheduled {
		source = "scheduled"
	}
	result := &service.CandyMonitorResult{AccountID: accountID, ModelID: model, Source: source, Verdict: "running", Expected: 21}
	if err = tx.QueryRowContext(ctx, `INSERT INTO candy_monitor_results(account_id,model_id,source) VALUES($1,$2,$3) RETURNING id,started_at`, accountID, model, source).Scan(&result.ID, &result.StartedAt); err != nil {
		return nil, err
	}
	if _, err = tx.ExecContext(ctx, `UPDATE candy_monitor_accounts SET running_result_id=$2 WHERE account_id=$1`, accountID, result.ID); err != nil {
		return nil, err
	}
	if err = tx.Commit(); err != nil {
		return nil, err
	}
	return result, nil
}
func (r *candyMonitorRepository) Finish(ctx context.Context, v *service.CandyMonitorResult) error {
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	// Lock and fence against a process that resumed after its lease expired.
	var current sql.NullInt64
	if err = tx.QueryRowContext(ctx, `SELECT running_result_id FROM candy_monitor_accounts WHERE account_id=$1 FOR UPDATE`, v.AccountID).Scan(&current); err != nil {
		return err
	}
	if !current.Valid || current.Int64 != v.ID {
		return nil
	}
	updated, err := tx.ExecContext(ctx, `UPDATE candy_monitor_results SET verdict=$2,reason=$3,actual=$4,expected=$5,duration_ms=$6,response_text=$7,error_message=$8,finished_at=$9 WHERE id=$1 AND verdict='running'`, v.ID, v.Verdict, v.Reason, v.Actual, v.Expected, v.DurationMs, v.ResponseText, v.ErrorMessage, v.FinishedAt)
	if err != nil {
		return err
	}
	n, err := updated.RowsAffected()
	if err != nil {
		return err
	}
	if n > 0 {
		if _, err = tx.ExecContext(ctx, `UPDATE candy_monitor_accounts SET total_tests=total_tests+1,
 answer_21_count=answer_21_count+CASE WHEN $2='pass' AND $3::bigint=21 THEN 1 ELSE 0 END,
 answer_29_count=answer_29_count+CASE WHEN $2='incorrect' AND $3::bigint=29 THEN 1 ELSE 0 END,
 other_answer_count=other_answer_count+CASE WHEN $2='incorrect' AND $3::bigint<>29 THEN 1 ELSE 0 END,
 inconclusive_count=inconclusive_count+CASE WHEN $2 IN ('inconclusive','invalid_format') THEN 1 ELSE 0 END,
 last_valid_answer=CASE WHEN $2 IN ('pass','incorrect') AND $3::bigint IS NOT NULL THEN $3::bigint ELSE last_valid_answer END,
 last_valid_at=CASE WHEN $2 IN ('pass','incorrect') AND $3::bigint IS NOT NULL THEN $4 ELSE last_valid_at END
 WHERE account_id=$1`, v.AccountID, v.Verdict, v.Actual, v.StartedAt); err != nil {
			return err
		}
	}
	if _, err = tx.ExecContext(ctx, `UPDATE candy_monitor_accounts SET lease_until=NULL,running_result_id=NULL WHERE account_id=$1 AND running_result_id=$2`, v.AccountID, v.ID); err != nil {
		return err
	}
	if _, err = tx.ExecContext(ctx, `DELETE FROM candy_monitor_results WHERE account_id=$1 AND verdict<>'running' AND id NOT IN (SELECT id FROM candy_monitor_results WHERE account_id=$1 ORDER BY id DESC LIMIT (SELECT max_results FROM candy_monitor_settings WHERE singleton))`, v.AccountID); err != nil {
		return err
	}
	return tx.Commit()
}

const candyResultColumns = `id,account_id,model_id,source,verdict,reason,actual,expected,duration_ms,response_text,error_message,started_at,finished_at`

func scanCandyResult(row scannable) (*service.CandyMonitorResult, error) {
	v := &service.CandyMonitorResult{}
	err := row.Scan(&v.ID, &v.AccountID, &v.ModelID, &v.Source, &v.Verdict, &v.Reason, &v.Actual, &v.Expected, &v.DurationMs, &v.ResponseText, &v.ErrorMessage, &v.StartedAt, &v.FinishedAt)
	return v, err
}
func (r *candyMonitorRepository) Result(ctx context.Context, id int64) (*service.CandyMonitorResult, error) {
	return scanCandyResult(r.db.QueryRowContext(ctx, `SELECT `+candyResultColumns+` FROM candy_monitor_results WHERE id=$1`, id))
}
func (r *candyMonitorRepository) History(ctx context.Context, id int64, limit int) ([]service.CandyMonitorResult, error) {
	if limit < 1 || limit > 100 {
		limit = 50
	}
	rows, err := r.db.QueryContext(ctx, `SELECT `+candyResultColumns+` FROM candy_monitor_results WHERE account_id=$1 ORDER BY id DESC LIMIT $2`, id, limit)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	out := make([]service.CandyMonitorResult, 0)
	for rows.Next() {
		v, err := scanCandyResult(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, *v)
	}
	return out, rows.Err()
}
