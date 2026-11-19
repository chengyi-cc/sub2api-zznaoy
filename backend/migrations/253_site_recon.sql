-- 站点对账（fork 自有模块）：本站收入 vs 上游成本的每日对账结果、运行记录、NewAPI 上游访问令牌。
-- 只新增表，不改动任何现有表。日期 / 时间按 TEXT 存（YYYY-MM-DD / RFC3339），与对账逻辑一致。

CREATE TABLE IF NOT EXISTS site_recon_daily (
    day           TEXT             NOT NULL,
    account_id    BIGINT           NOT NULL,
    account_name  TEXT             NOT NULL DEFAULT '',
    upstream      TEXT             NOT NULL DEFAULT '',
    unit_key      TEXT             NOT NULL DEFAULT '',
    our_requests  BIGINT           NOT NULL DEFAULT 0,
    up_requests   BIGINT,
    our_list_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
    up_list_cost  DOUBLE PRECISION,
    revenue       DOUBLE PRECISION NOT NULL DEFAULT 0,
    cost          DOUBLE PRECISION,
    up_balance    DOUBLE PRECISION,
    scanned_rows  BIGINT,
    anomalies     BIGINT,
    mismatches    BIGINT,
    error_text    TEXT,
    updated_at    TEXT             NOT NULL DEFAULT '',
    PRIMARY KEY (day, account_id)
);
CREATE INDEX IF NOT EXISTS idx_site_recon_daily_account ON site_recon_daily (account_id, day);
CREATE INDEX IF NOT EXISTS idx_site_recon_daily_unit ON site_recon_daily (unit_key, day);

CREATE TABLE IF NOT EXISTS site_recon_models (
    day           TEXT             NOT NULL,
    account_id    BIGINT           NOT NULL,
    model         TEXT             NOT NULL,
    our_requests  BIGINT           NOT NULL DEFAULT 0,
    up_requests   BIGINT,
    revenue       DOUBLE PRECISION NOT NULL DEFAULT 0,
    cost          DOUBLE PRECISION,
    our_list_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
    PRIMARY KEY (day, account_id, model)
);

CREATE TABLE IF NOT EXISTS site_recon_groups (
    day             TEXT             NOT NULL,
    account_id      BIGINT           NOT NULL,
    group_name      TEXT             NOT NULL,
    rate_multiplier DOUBLE PRECISION NOT NULL,
    requests        BIGINT           NOT NULL DEFAULT 0,
    list_cost       DOUBLE PRECISION NOT NULL DEFAULT 0,
    revenue         DOUBLE PRECISION NOT NULL DEFAULT 0,
    cost            DOUBLE PRECISION,
    PRIMARY KEY (day, account_id, group_name, rate_multiplier)
);

-- 异常样本（每个账号每天每类最多留一批，够定位就行）
CREATE TABLE IF NOT EXISTS site_recon_anomalies (
    day         TEXT             NOT NULL,
    account_id  BIGINT           NOT NULL,
    log_id      BIGINT           NOT NULL,
    kind        TEXT             NOT NULL,
    created_at  TEXT             NOT NULL DEFAULT '',
    user_id     BIGINT           NOT NULL DEFAULT 0,
    model       TEXT             NOT NULL DEFAULT '',
    total_cost  DOUBLE PRECISION NOT NULL DEFAULT 0,
    actual_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
    tokens      BIGINT           NOT NULL DEFAULT 0,
    PRIMARY KEY (day, account_id, log_id)
);

-- 异常汇总（全量计数）。detail：模型不一致时是“上游实际返回的模型”，其余为空
CREATE TABLE IF NOT EXISTS site_recon_anomaly_stats (
    day         TEXT             NOT NULL,
    account_id  BIGINT           NOT NULL,
    kind        TEXT             NOT NULL,
    model       TEXT             NOT NULL,
    detail      TEXT             NOT NULL,
    requests    BIGINT           NOT NULL DEFAULT 0,
    list_cost   DOUBLE PRECISION NOT NULL DEFAULT 0,
    actual_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
    PRIMARY KEY (day, account_id, kind, model, detail)
);

CREATE TABLE IF NOT EXISTS site_recon_runs (
    id           BIGSERIAL PRIMARY KEY,
    trigger_kind TEXT NOT NULL,
    target       TEXT NOT NULL DEFAULT '',
    days         INT  NOT NULL DEFAULT 0,
    started_at   TEXT NOT NULL,
    finished_at  TEXT NOT NULL DEFAULT '',
    heartbeat_at TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL,
    summary      TEXT NOT NULL DEFAULT '{}',
    log          TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_site_recon_runs_status ON site_recon_runs (status, heartbeat_at);

-- NewAPI 上游后台访问令牌：只存密文（SecretEncryptor 加密），接口只回显尾号。
CREATE TABLE IF NOT EXISTS site_recon_upstream_tokens (
    host             TEXT PRIMARY KEY,
    token_ciphertext TEXT   NOT NULL,
    token_hint       TEXT   NOT NULL DEFAULT '',
    updated_at       TEXT   NOT NULL DEFAULT '',
    updated_by       BIGINT NOT NULL DEFAULT 0
);
