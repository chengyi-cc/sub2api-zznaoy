ALTER TABLE candy_monitor_accounts
    ADD COLUMN total_tests BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN answer_21_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN answer_29_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN other_answer_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN inconclusive_count BIGINT NOT NULL DEFAULT 0;

-- Backfill retained history once; future counters survive history retention.
UPDATE candy_monitor_accounts m SET
    total_tests=c.total,
    answer_21_count=c.answer21,
    answer_29_count=c.answer29,
    other_answer_count=c.other,
    inconclusive_count=c.inconclusive
FROM (
    SELECT account_id, COUNT(*) AS total,
        COUNT(*) FILTER (WHERE verdict='pass' AND actual=21) AS answer21,
        COUNT(*) FILTER (WHERE verdict='incorrect' AND actual=29) AS answer29,
        COUNT(*) FILTER (WHERE verdict='incorrect' AND actual<>29) AS other,
        COUNT(*) FILTER (WHERE verdict IN ('inconclusive','invalid_format')) AS inconclusive
    FROM candy_monitor_results WHERE verdict<>'running' GROUP BY account_id
) c WHERE m.account_id=c.account_id;
