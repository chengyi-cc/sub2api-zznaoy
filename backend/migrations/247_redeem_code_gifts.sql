-- Gift identity is independent of editable notes. Existing codes remain paid codes.
ALTER TABLE redeem_codes
    ADD COLUMN IF NOT EXISTS is_gift BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS gift_pair_id VARCHAR(32);

CREATE UNIQUE INDEX IF NOT EXISTS redeemcode_gift_pair_id_is_gift
    ON redeem_codes (gift_pair_id, is_gift);

ALTER TABLE redeem_codes ADD CONSTRAINT redeem_codes_gift_valid CHECK (
    NOT is_gift OR (type = 'balance' AND value > 0 AND value < 1000000000000 AND gift_pair_id IS NOT NULL AND length(gift_pair_id) = 32)
);

-- Ordinary status/notes/expiry updates cannot turn a gift into paid recharge.
CREATE OR REPLACE FUNCTION protect_redeem_code_gift_identity() RETURNS trigger AS $$
BEGIN
    IF NEW.is_gift IS DISTINCT FROM OLD.is_gift
       OR NEW.gift_pair_id IS DISTINCT FROM OLD.gift_pair_id THEN
        RAISE EXCEPTION 'redeem code gift identity is immutable';
    END IF;
    IF OLD.gift_pair_id IS NOT NULL AND (
        NEW.code IS DISTINCT FROM OLD.code OR NEW.type IS DISTINCT FROM OLD.type
        OR NEW.value IS DISTINCT FROM OLD.value
    ) THEN
        RAISE EXCEPTION 'paired redeem code value and type are immutable';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER redeem_code_gift_identity_immutable
    BEFORE UPDATE ON redeem_codes
    FOR EACH ROW EXECUTE FUNCTION protect_redeem_code_gift_identity();
