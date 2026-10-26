-- A paired code that has been redeemed must never become redeemable again.
-- Keep this separate from migration 247 for installations that applied 247.
CREATE OR REPLACE FUNCTION protect_redeem_code_gift_consumption() RETURNS trigger AS $$
BEGIN
    IF OLD.gift_pair_id IS NOT NULL AND
       (OLD.status = 'used' OR OLD.used_at IS NOT NULL OR OLD.used_by IS NOT NULL) THEN
        IF NEW.status IS DISTINCT FROM 'used' OR NEW.used_at IS DISTINCT FROM OLD.used_at THEN
            RAISE EXCEPTION 'redeemed paired code cannot be reactivated';
        END IF;
    END IF;
    -- used_by can be cleared by ON DELETE SET NULL when a user is deleted;
    -- the status and timestamp still permanently prevent a second redemption.
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER redeem_code_gift_consumption_immutable
    BEFORE UPDATE ON redeem_codes
    FOR EACH ROW EXECUTE FUNCTION protect_redeem_code_gift_consumption();
