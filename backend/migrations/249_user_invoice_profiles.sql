-- 用户固定开票资料（仅企业抬头）
-- 用户在个人资料页绑定一次后锁定，仅管理员可修改或重置（删除行）。
-- 提交发票申请时后端从此表取抬头/税号快照写入 invoice_requests，不再接受前端传入。

CREATE TABLE IF NOT EXISTS user_invoice_profiles (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title       VARCHAR(200) NOT NULL,
    tax_no      VARCHAR(50) NOT NULL,
    updated_by  BIGINT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS invoiceprofile_user_id ON user_invoice_profiles(user_id);
CREATE INDEX IF NOT EXISTS invoiceprofile_tax_no ON user_invoice_profiles(tax_no);
