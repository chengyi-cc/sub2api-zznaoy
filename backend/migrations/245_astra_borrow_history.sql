-- Fixed-egress borrowing audit only. Never store cookies, tickets or credentials.
CREATE TABLE IF NOT EXISTS astra_borrow_history (
    id BIGSERIAL PRIMARY KEY,
    source_account_id BIGINT NOT NULL DEFAULT 0,
    target_account_id BIGINT NOT NULL DEFAULT 0,
    passed BOOLEAN NOT NULL,
    reason VARCHAR(80) NOT NULL,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_astra_borrow_history_checked ON astra_borrow_history (checked_at);
