package repository

import (
	"context"
	"fmt"
	"github.com/Wei-Shaw/sub2api/internal/service"
)

var _ service.AstraBorrowStore = (*settingRepository)(nil)

func (r *settingRepository) AppendAstraBorrowHistory(ctx context.Context, row service.AstraBorrowHistory) error {
	if len(row.Reason) > 80 {
		return fmt.Errorf("invalid Astra history reason")
	}
	_, err := r.client.ExecContext(ctx, `INSERT INTO astra_borrow_history (source_account_id,target_account_id,passed,reason,checked_at) VALUES ($1,$2,$3,$4,$5)`, row.SourceAccountID, row.TargetAccountID, row.Passed, row.Reason, row.CheckedAt)
	if err != nil {
		return err
	}
	// Thirty-day retention, bounded deletion work on each insertion.
	_, err = r.client.ExecContext(ctx, `DELETE FROM astra_borrow_history WHERE id IN (SELECT id FROM astra_borrow_history WHERE checked_at < NOW() - INTERVAL '30 days' ORDER BY checked_at LIMIT 200)`)
	return err
}

func (r *settingRepository) ListAstraBorrowHistory(ctx context.Context, before int64, limit int) ([]service.AstraBorrowHistory, error) {
	if before < 0 || limit < 1 || limit > 100 {
		return nil, fmt.Errorf("invalid Astra history query")
	}
	rows, err := r.client.QueryContext(ctx, `SELECT id,source_account_id,target_account_id,passed,reason,checked_at FROM astra_borrow_history WHERE ($1::bigint=0 OR id<$1) ORDER BY id DESC LIMIT $2`, before, limit)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	items := []service.AstraBorrowHistory{}
	for rows.Next() {
		var item service.AstraBorrowHistory
		if err := rows.Scan(&item.ID, &item.SourceAccountID, &item.TargetAccountID, &item.Passed, &item.Reason, &item.CheckedAt); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}
