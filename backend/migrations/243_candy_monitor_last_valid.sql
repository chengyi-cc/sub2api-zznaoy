ALTER TABLE candy_monitor_accounts
    ADD COLUMN last_valid_answer BIGINT,
    ADD COLUMN last_valid_at TIMESTAMPTZ;

-- Keep the last parsed answer independently of transient failures and retention.
UPDATE candy_monitor_accounts m SET
    last_valid_answer=r.actual, last_valid_at=r.started_at
FROM (
    SELECT DISTINCT ON (account_id) account_id, actual, started_at
    FROM candy_monitor_results
    WHERE verdict IN ('pass','incorrect') AND actual IS NOT NULL
    ORDER BY account_id, id DESC
) r WHERE m.account_id=r.account_id;
