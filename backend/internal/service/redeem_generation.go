package service

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"time"

	infraerrors "github.com/Wei-Shaw/sub2api/internal/pkg/errors"
	"github.com/shopspring/decimal"
)

// GiftPairRepository enables recovery when codes committed but the response or
// idempotency result could not be saved. A unique pair identity prevents reminting.
type GiftPairRepository interface {
	FindGiftPairs(ctx context.Context, pairIDs []string) ([]RedeemCode, error)
}

// GenerateRedeemCodes accepts only an opt-in flag. Gift amounts and identities
// are generated on the server and saved in the same transaction as paid codes.
func (s *adminServiceImpl) GenerateRedeemCodes(ctx context.Context, input *GenerateRedeemCodesInput) ([]RedeemCode, error) {
	if input == nil || input.Count < 1 || input.Count > 500 {
		return nil, infraerrors.BadRequest("REDEEM_CODE_COUNT_INVALID", "count must be between 1 and 500")
	}
	switch input.Type {
	case RedeemTypeBalance, RedeemTypeConcurrency, RedeemTypeSubscription, RedeemTypeInvitation:
	default:
		return nil, infraerrors.BadRequest("REDEEM_CODE_TYPE_INVALID", "invalid redeem code type")
	}
	// Reject non-finite, out-of-range and silently rounded monetary values.
	// The database column is decimal(20,8), so it has 12 integer digits.
	if math.IsNaN(input.Value) || math.IsInf(input.Value, 0) || math.Abs(input.Value) >= 1e12 {
		return nil, infraerrors.BadRequest("REDEEM_CODE_VALUE_INVALID", "value is outside the supported range")
	}
	value := decimal.NewFromFloat(input.Value)
	if !value.Equal(value.Round(8)) || (input.Type != RedeemTypeInvitation && input.Type != RedeemTypeSubscription && value.IsZero()) {
		return nil, infraerrors.BadRequest("REDEEM_CODE_VALUE_INVALID", "value must be nonzero with at most 8 decimal places")
	}
	var giftValue float64
	if input.IncludeGift {
		if input.Type != RedeemTypeBalance || !value.IsPositive() || !value.Equal(value.Round(2)) {
			return nil, infraerrors.BadRequest("REDEEM_CODE_GIFT_INVALID", "gifts require a positive balance amount with at most 2 decimal places")
		}
		// Decimal arithmetic: 10 -> 0.1; 0.01 -> 0.0001, with no rounding up.
		giftValue, _ = value.Shift(-2).Float64()
	}
	if input.ExpiresAt != nil && !input.ExpiresAt.After(time.Now()) {
		return nil, ErrRedeemCodeExpired
	}
	if input.Type == RedeemTypeSubscription {
		if input.GroupID == nil {
			return nil, infraerrors.BadRequest("REDEEM_CODE_GROUP_REQUIRED", "group_id is required for subscription type")
		}
		group, err := s.groupRepo.GetByID(ctx, *input.GroupID)
		if err != nil {
			return nil, fmt.Errorf("group not found: %w", err)
		}
		if !group.IsSubscriptionType() {
			return nil, infraerrors.BadRequest("REDEEM_CODE_GROUP_INVALID", "group must be subscription type")
		}
	}
	var pairIDs []string
	var giftRepo GiftPairRepository
	if input.IncludeGift && input.OperationKey != "" {
		var ok bool
		giftRepo, ok = s.redeemCodeRepo.(GiftPairRepository)
		if !ok {
			return nil, infraerrors.InternalServer("REDEEM_CODE_GIFT_RECOVERY_UNAVAILABLE", "gift generation recovery is unavailable")
		}
		for i := 0; i < input.Count; i++ {
			hash := sha256.Sum256([]byte(fmt.Sprintf("%s:%d", input.OperationKey, i)))
			pairIDs = append(pairIDs, hex.EncodeToString(hash[:16]))
		}
		existing, err := recoverGiftPairs(ctx, giftRepo, pairIDs, input.Value)
		if err != nil {
			return nil, err
		}
		if existing != nil {
			return existing, nil
		}
	}
	capacity := input.Count
	if input.IncludeGift {
		capacity *= 2
	}
	codes := make([]RedeemCode, 0, capacity)
	for i := 0; i < input.Count; i++ {
		codeValue, err := GenerateRedeemCode()
		if err != nil {
			return nil, err
		}
		code := RedeemCode{
			Code: codeValue, Type: input.Type, Value: input.Value,
			Status: StatusUnused, ExpiresAt: input.ExpiresAt,
		}
		if input.Type == RedeemTypeInvitation {
			code.Value = 0
		}
		if input.Type == RedeemTypeSubscription {
			code.GroupID = input.GroupID
			code.ValidityDays = input.ValidityDays
			if code.ValidityDays <= 0 {
				code.ValidityDays = 30
			}
		}
		if input.IncludeGift {
			// This identifier is independent of both redeemable secrets and is
			// returned only to admins. Knowing one code never reveals the other.
			var pairID string
			if pairIDs != nil {
				pairID = pairIDs[i]
			} else {
				pairID, err = GenerateRedeemCode()
				if err != nil {
					return nil, err
				}
			}
			giftCode, err := GenerateRedeemCode()
			if err != nil {
				return nil, err
			}
			code.GiftPairID = &pairID
			codes = append(codes, code, RedeemCode{
				Code: giftCode, Type: RedeemTypeBalance, Value: giftValue,
				IsGift: true, GiftPairID: &pairID, Status: StatusUnused,
				ExpiresAt: input.ExpiresAt,
			})
		} else {
			codes = append(codes, code)
		}
	}
	if err := s.redeemCodeRepo.CreateBatch(ctx, codes); err != nil {
		// A concurrent retry may have won the unique pair insert. Never mint
		// another batch just because the response cache was unavailable.
		if giftRepo != nil {
			existing, recoverErr := recoverGiftPairs(ctx, giftRepo, pairIDs, input.Value)
			if recoverErr != nil {
				return nil, recoverErr
			}
			if existing != nil {
				return existing, nil
			}
		}
		return nil, err
	}
	return codes, nil
}

func recoverGiftPairs(ctx context.Context, repo GiftPairRepository, pairIDs []string, value float64) ([]RedeemCode, error) {
	existing, err := repo.FindGiftPairs(ctx, pairIDs)
	if err != nil {
		return nil, err
	}
	if len(existing) == 0 {
		return nil, nil
	}
	invalid := func() ([]RedeemCode, error) {
		return nil, infraerrors.Conflict("REDEEM_CODE_GIFT_BATCH_CONFLICT", "gift generation batch changed or is incomplete; review existing codes")
	}
	if len(existing) != len(pairIDs)*2 {
		return invalid()
	}
	byPair := make(map[string][2]*RedeemCode, len(pairIDs))
	giftValue, _ := decimal.NewFromFloat(value).Shift(-2).Float64()
	for i := range existing {
		code := &existing[i]
		if code.GiftPairID == nil || code.Type != RedeemTypeBalance {
			return invalid()
		}
		index := 0
		expected := value
		if code.IsGift {
			index = 1
			expected = giftValue
		}
		if code.Value != expected {
			return invalid()
		}
		pair := byPair[*code.GiftPairID]
		if pair[index] != nil {
			return invalid()
		}
		pair[index] = code
		byPair[*code.GiftPairID] = pair
	}
	result := make([]RedeemCode, 0, len(existing))
	for _, id := range pairIDs {
		pair := byPair[id]
		if pair[0] == nil || pair[1] == nil {
			return invalid()
		}
		result = append(result, *pair[0], *pair[1])
	}
	return result, nil
}
