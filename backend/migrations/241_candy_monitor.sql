CREATE TABLE IF NOT EXISTS candy_monitor_settings (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    model_id VARCHAR(200) NOT NULL DEFAULT 'gpt-6-astra',
    interval_minutes INT NOT NULL DEFAULT 60 CHECK (interval_minutes BETWEEN 5 AND 10080),
    max_results INT NOT NULL DEFAULT 50 CHECK (max_results BETWEEN 10 AND 500)
);
INSERT INTO candy_monitor_settings (singleton) VALUES (TRUE) ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS candy_monitor_accounts (
    account_id BIGINT PRIMARY KEY REFERENCES accounts(id) ON DELETE CASCADE,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    use_defaults BOOLEAN NOT NULL DEFAULT TRUE,
    model_id VARCHAR(200) NOT NULL DEFAULT 'gpt-6-astra',
    interval_minutes INT NOT NULL DEFAULT 60 CHECK (interval_minutes BETWEEN 5 AND 10080),
    next_run_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_run_at TIMESTAMPTZ,
    lease_until TIMESTAMPTZ,
    running_result_id BIGINT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_candy_monitor_due ON candy_monitor_accounts(next_run_at) WHERE enabled;

CREATE TABLE IF NOT EXISTS candy_monitor_results (
    id BIGSERIAL PRIMARY KEY,
    account_id BIGINT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    model_id VARCHAR(200) NOT NULL,
    source VARCHAR(20) NOT NULL,
    verdict VARCHAR(30) NOT NULL DEFAULT 'running',
    reason VARCHAR(100) NOT NULL DEFAULT '',
    actual BIGINT,
    expected INT NOT NULL DEFAULT 21,
    duration_ms BIGINT NOT NULL DEFAULT 0,
    response_text TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_candy_monitor_results_account ON candy_monitor_results(account_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_candy_monitor_running ON candy_monitor_results(started_at) WHERE verdict = 'running';
