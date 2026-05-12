-- 为 invoice_requests 增加 redeem_code_ids 字段以支持兑换码开票
-- （与 payment_order_ids 可混合合并到同一张发票）

ALTER TABLE invoice_requests
    ADD COLUMN IF NOT EXISTS redeem_code_ids JSONB NOT NULL DEFAULT '[]'::jsonb;

-- GIN 索引用于"兑换码是否已被发票申请占用"反向查询：
--   SELECT 1 FROM invoice_requests
--   WHERE redeem_code_ids @> '[<code_id>]'
--     AND status IN ('pending','approved','issued')
CREATE INDEX IF NOT EXISTS idx_invoice_requests_redeem_code_ids
    ON invoice_requests USING GIN (redeem_code_ids);