package repository

import (
	"context"
	"database/sql"
	"encoding/json"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/service"
)

// Protocol changes and scheduler invalidation commit with the current leased verdict.
func enableExcelAfterCandyFailure(ctx context.Context, tx *sql.Tx, v *service.CandyMonitorResult) error {
	if v.Verdict != "incorrect" || v.Actual == nil {
		return nil
	}
	var enabled bool
	var credentials, extra []byte
	a := &service.Account{ID: v.AccountID}
	err := tx.QueryRowContext(ctx, `SELECT a.platform,a.type,a.parent_account_id,a.credentials,COALESCE(a.extra,'{}'::jsonb),
 COALESCE(m.auto_excel_on_incorrect,s.auto_excel_on_incorrect)
 FROM accounts a JOIN candy_monitor_accounts m ON m.account_id=a.id CROSS JOIN candy_monitor_settings s
 WHERE a.id=$1 AND a.deleted_at IS NULL FOR UPDATE OF a`, v.AccountID).Scan(&a.Platform, &a.Type, &a.ParentAccountID, &credentials, &extra, &enabled)
	if err == sql.ErrNoRows {
		return nil
	}
	if err != nil {
		return err
	}
	if !enabled {
		return nil
	}
	if err = json.Unmarshal(credentials, &a.Credentials); err != nil {
		return err
	}
	if err = json.Unmarshal(extra, &a.Extra); err != nil {
		return err
	}
	if a.Extra == nil {
		a.Extra = map[string]any{}
	}
	if a.Extra["openai_excel_bps"] == true {
		return nil
	}
	if event, ok := a.Extra["openai_excel_bps_last_transition"].(map[string]any); ok {
		if event["reason"] == "upstream_403" {
			return nil
		}
		if at, ok := event["at"].(string); ok {
			when, e := time.Parse(time.RFC3339Nano, at)
			if e == nil && when.After(v.StartedAt) {
				return nil
			}
		}
	}
	a.Extra["openai_excel_bps"] = true
	if !a.IsExcelBPSEnabled() {
		return nil
	}
	a.Extra["openai_rpm_overflow"] = true
	a.Extra["openai_excel_bps_cache_creation_as_input"] = true
	a.Extra["openai_excel_bps_auto_disable_on_403"] = true
	if a.GetBaseRPM() <= 0 {
		a.Extra["base_rpm"] = 15
	}
	models := a.ExcelBPSModels()
	model := a.GetMappedModel(v.ModelID)
	found := false
	for _, m := range models {
		if m == model {
			found = true
		}
	}
	if !found {
		models = append(models, model)
	}
	a.Extra["openai_excel_bps_models"] = models
	a.Extra = service.ExcelBPSActivationExtra(a.Extra, false, "candy_incorrect")
	payload, err := json.Marshal(a.Extra)
	if err != nil {
		return err
	}
	if _, err = tx.ExecContext(ctx, `UPDATE accounts SET extra=$2::jsonb,updated_at=NOW() WHERE id=$1`, a.ID, string(payload)); err != nil {
		return err
	}
	return enqueueSchedulerOutbox(ctx, tx, service.SchedulerOutboxEventAccountChanged, &a.ID, nil, nil)
}
