-- 发票申请表
-- 用户提交开票申请 → 管理员审核（通过/驳回）→ 通过后管理员开票（上传 PDF 填发票号）
-- PDF 文件路径仅作为内部引用，不直接暴露公网，必须通过鉴权下载接口访问

CREATE TABLE IF NOT EXISTS invoice_requests (
    id                 BIGSERIAL PRIMARY KEY,
    user_id            BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    payment_order_ids  JSONB NOT NULL DEFAULT '[]'::jsonb,
    amount             DECIMAL(20, 2) NOT NULL DEFAULT 0,
    invoice_type       VARCHAR(20) NOT NULL DEFAULT 'personal',
    title              VARCHAR(200) NOT NULL,
    tax_no             VARCHAR(50),
    recipient_email    VARCHAR(255),
    remark             TEXT,
    status             VARCHAR(20) NOT NULL DEFAULT 'pending',
    reject_reason      TEXT,
    invoice_no         VARCHAR(100),
    invoice_file_path  VARCHAR(500),
    issued_at          TIMESTAMPTZ,
    processed_by       BIGINT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_invoice_requests_user_id ON invoice_requests(user_id);
CREATE INDEX IF NOT EXISTS idx_invoice_requests_status ON invoice_requests(status);
CREATE INDEX IF NOT EXISTS idx_invoice_requests_created_at ON invoice_requests(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_invoice_requests_user_status ON invoice_requests(user_id, status);

-- GIN 索引用于"订单是否已被发票申请占用"的反向查询：
-- SELECT 1 FROM invoice_requests WHERE payment_order_ids @> '[<order_id>]' AND status IN ('pending','approved','issued')
CREATE INDEX IF NOT EXISTS idx_invoice_requests_payment_order_ids
    ON invoice_requests USING GIN (payment_order_ids);