-- Adapted from ranxi2001/sub2api 254_openai_oauth_reauth.sql
-- OpenAI OAuth re-authentication configuration and durable task state.
-- The OTP endpoint is encrypted by the application before it reaches this table.
CREATE TABLE IF NOT EXISTS openai_oauth_reauth_configs (
    account_id BIGINT PRIMARY KEY REFERENCES accounts(id) ON DELETE CASCADE,
    login_email TEXT NOT NULL,
    otp_url_ciphertext TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS openai_oauth_reauth_tasks (
    id BIGSERIAL PRIMARY KEY,
    account_id BIGINT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'queued',
    stage TEXT NOT NULL DEFAULT 'queued',
    worker_id TEXT,
    auth_session_id TEXT,
    expected_credentials_hash TEXT NOT NULL,
    error_message TEXT,
    attempt INTEGER NOT NULL DEFAULT 0,
    claimed_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT openai_oauth_reauth_tasks_status_check CHECK (
        status IN ('queued', 'running', 'callback_processing', 'succeeded', 'failed')
    ),
    CONSTRAINT openai_oauth_reauth_tasks_stage_check CHECK (
        stage IN (
            'queued', 'starting', 'protocol_connecting', 'email_submitted',
            'waiting_otp', 'otp_submitted', 'waiting_callback',
            'exchanging_token', 'applying_credentials', 'succeeded', 'failed'
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_openai_oauth_reauth_tasks_account_created
    ON openai_oauth_reauth_tasks (account_id, created_at DESC, id DESC);

CREATE UNIQUE INDEX IF NOT EXISTS uq_openai_oauth_reauth_tasks_active_account
    ON openai_oauth_reauth_tasks (account_id)
    WHERE status IN ('queued', 'running', 'callback_processing');

CREATE INDEX IF NOT EXISTS idx_openai_oauth_reauth_tasks_claimable
    ON openai_oauth_reauth_tasks (status, created_at, id);


-- Adapted from ranxi2001/sub2api 255_account_token_guard_v2.sql
-- Credential Guard V2: per-account encrypted login methods and durable probe leases.
ALTER TABLE openai_oauth_reauth_configs
    ADD COLUMN IF NOT EXISTS credential_mode TEXT NOT NULL DEFAULT 'email_otp_url',
    ADD COLUMN IF NOT EXISTS password_ciphertext TEXT,
    ADD COLUMN IF NOT EXISTS totp_secret_ciphertext TEXT;

ALTER TABLE openai_oauth_reauth_configs
    ALTER COLUMN otp_url_ciphertext DROP NOT NULL;

UPDATE openai_oauth_reauth_configs
SET credential_mode = 'email_otp_url'
WHERE credential_mode IS NULL OR BTRIM(credential_mode) = '';

ALTER TABLE openai_oauth_reauth_configs
    DROP CONSTRAINT IF EXISTS openai_oauth_reauth_configs_credential_mode_check;
ALTER TABLE openai_oauth_reauth_configs
    ADD CONSTRAINT openai_oauth_reauth_configs_credential_mode_check CHECK (
        credential_mode IN ('email_otp_url', 'password_totp')
    );

ALTER TABLE openai_oauth_reauth_tasks
    DROP CONSTRAINT IF EXISTS openai_oauth_reauth_tasks_stage_check;
ALTER TABLE openai_oauth_reauth_tasks
    ADD CONSTRAINT openai_oauth_reauth_tasks_stage_check CHECK (
        stage IN (
            'queued', 'starting', 'protocol_connecting', 'email_submitted',
            'waiting_otp', 'otp_submitted', 'password_submitted', 'mfa_submitted',
            'waiting_callback', 'exchanging_token', 'applying_credentials',
            'succeeded', 'failed'
        )
    );

CREATE TABLE IF NOT EXISTS account_token_guard_v2_accounts (
    account_id BIGINT PRIMARY KEY REFERENCES accounts(id) ON DELETE CASCADE,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    auto_relogin_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    probe_state TEXT NOT NULL DEFAULT 'pending',
    probe_detail TEXT NOT NULL DEFAULT '',
    fail_streak INTEGER NOT NULL DEFAULT 0,
    last_probe_at TIMESTAMPTZ,
    last_reauth_at TIMESTAMPTZ,
    next_probe_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cooldown_until TIMESTAMPTZ,
    blocked_reason TEXT NOT NULL DEFAULT '',
    lease_owner TEXT,
    lease_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT account_token_guard_v2_probe_state_check CHECK (
        probe_state IN ('pending', 'ok', 'auth', 'transient')
    ),
    CONSTRAINT account_token_guard_v2_fail_streak_check CHECK (fail_streak >= 0)
);

CREATE INDEX IF NOT EXISTS idx_account_token_guard_v2_due
    ON account_token_guard_v2_accounts (next_probe_at, account_id)
    WHERE enabled = TRUE;

CREATE INDEX IF NOT EXISTS idx_account_token_guard_v2_lease
    ON account_token_guard_v2_accounts (lease_until)
    WHERE lease_until IS NOT NULL;


-- Adapted from ranxi2001/sub2api 256_openai_oauth_reauth_proxy_override.sql
-- Optional managed proxy used only by OpenAI OAuth re-login.
ALTER TABLE openai_oauth_reauth_configs
    ADD COLUMN IF NOT EXISTS proxy_id BIGINT REFERENCES proxies(id) ON DELETE SET NULL;


-- Adapted from ranxi2001/sub2api 257_openai_oauth_reauth_proxy_source.sql
-- Distinguish account routing, a selected managed proxy, and the managed Mihomo pool.
ALTER TABLE openai_oauth_reauth_configs
    ADD COLUMN IF NOT EXISTS proxy_source TEXT;

UPDATE openai_oauth_reauth_configs
SET proxy_source = CASE
    WHEN proxy_id IS NOT NULL THEN 'managed_proxy'
    ELSE 'account'
END
WHERE proxy_source IS NULL OR BTRIM(proxy_source) = '';

ALTER TABLE openai_oauth_reauth_configs
    ALTER COLUMN proxy_source SET DEFAULT 'account',
    ALTER COLUMN proxy_source SET NOT NULL;

ALTER TABLE openai_oauth_reauth_configs
    DROP CONSTRAINT IF EXISTS openai_oauth_reauth_configs_proxy_source_check;
ALTER TABLE openai_oauth_reauth_configs
    ADD CONSTRAINT openai_oauth_reauth_configs_proxy_source_check CHECK (
        (proxy_source = 'managed_proxy' AND proxy_id IS NOT NULL)
        OR (proxy_source IN ('account', 'mihomo') AND proxy_id IS NULL)
    );


-- Adapted from ranxi2001/sub2api 262_openai_oauth_reauth_engine.sql
ALTER TABLE openai_oauth_reauth_configs
    ADD COLUMN IF NOT EXISTS engine TEXT NOT NULL DEFAULT 'local_worker';

UPDATE openai_oauth_reauth_configs
SET engine = 'local_worker'
WHERE engine IS NULL OR BTRIM(engine) = '';

ALTER TABLE openai_oauth_reauth_configs
    DROP CONSTRAINT IF EXISTS openai_oauth_reauth_configs_engine_check;
ALTER TABLE openai_oauth_reauth_configs
    ADD CONSTRAINT openai_oauth_reauth_configs_engine_check CHECK (engine IN ('local_worker', 'session_studio'));
