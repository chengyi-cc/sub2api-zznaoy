//go:build unit

package service

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestAdminFulfillmentBypassesLimitAndKeepsRedeemAffiliate(t *testing.T) {
	ctx := context.Background()
	client := newPaymentConfigServiceTestClient(t)
	userID := int64(42)
	inviterID := int64(9001)
	code := &RedeemCode{
		ID: 102, Code: "ADMIN-SUCCESS", Type: RedeemTypeBalance, Value: 20, Status: StatusUnused,
	}
	redeemRepo := &paymentFulfillmentRedeemRepo{
		paymentOrderLifecycleRedeemRepo: paymentOrderLifecycleRedeemRepo{
			codesByCode: map[string]*RedeemCode{code.Code: code},
		},
	}
	userRepo := &mockUserRepo{getByIDUser: &User{ID: userID}}
	userRepo.updateBalanceFn = func(context.Context, int64, float64) error { return nil }
	cache := &paymentFulfillmentRedeemCacheStub{count: redeemMaxFailedAttempts}
	affiliateRepo := &paymentFulfillmentAffiliateRepoStub{
		inviteeSummary: &AffiliateSummary{
			UserID: userID, AffCode: "INVITEE", InviterID: &inviterID, CreatedAt: time.Now().Add(-time.Hour),
		},
		inviterSummary: &AffiliateSummary{
			UserID: inviterID, AffCode: "INVITER", CreatedAt: time.Now().Add(-2 * time.Hour),
		},
	}
	settingSvc := NewSettingService(&paymentFulfillmentSettingRepoStub{values: map[string]string{
		SettingKeyAffiliateEnabled:           "true",
		SettingKeyAffiliateRebateRate:        "10",
		SettingKeyAffiliateRebateFreezeHours: "0",
	}}, nil)
	affiliateSvc := NewAffiliateService(affiliateRepo, settingSvc, nil, nil)
	svc := NewRedeemService(redeemRepo, userRepo, nil, cache, nil, client, nil, affiliateSvc)

	result, err := svc.RedeemForAdminFulfillment(ctx, userID, code.Code)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Zero(t, cache.getCalls)
	require.Zero(t, cache.incrementCalls)
	require.Equal(t, 1, cache.acquireCalls)
	require.Equal(t, 1, cache.releaseCalls)
	require.Len(t, redeemRepo.useCalls, 1)
	require.Len(t, affiliateRepo.accrueCalls, 1)
	require.Equal(t, inviterID, affiliateRepo.accrueCalls[0].inviterID)
	require.Equal(t, userID, affiliateRepo.accrueCalls[0].inviteeUserID)
	require.InDelta(t, 2, affiliateRepo.accrueCalls[0].amount, 1e-8)
	require.Nil(t, affiliateRepo.accrueCalls[0].sourceOrderID)
}

type giftCreditUserRepo struct {
	*mockUserRepo
	giftCalls int
	gifted    float64
}

func (r *giftCreditUserRepo) CreditGiftBalance(_ context.Context, _ int64, amount float64) error {
	r.giftCalls++
	r.gifted += amount
	return nil
}

func TestGiftRedeemCreditsBalanceWithoutPaidRechargeOrRebate(t *testing.T) {
	ctx := context.Background()
	client := newPaymentConfigServiceTestClient(t)
	pairID := "0123456789abcdef0123456789abcdef"
	code := &RedeemCode{ID: 104, Code: "GIFT-SUCCESS", Type: RedeemTypeBalance, Value: 0.1, IsGift: true, GiftPairID: &pairID, Status: StatusUnused}
	repo := &paymentFulfillmentRedeemRepo{paymentOrderLifecycleRedeemRepo: paymentOrderLifecycleRedeemRepo{codesByCode: map[string]*RedeemCode{code.Code: code}}}
	userRepo := &giftCreditUserRepo{mockUserRepo: &mockUserRepo{getByIDUser: &User{ID: 42}}}
	// No paid UpdateBalance is permitted for a gift.
	userRepo.updateBalanceFn = func(context.Context, int64, float64) error { t.Fatal("gift counted as paid recharge"); return nil }
	inviterID := int64(9001)
	affiliateRepo := &paymentFulfillmentAffiliateRepoStub{
		inviteeSummary: &AffiliateSummary{UserID: 42, InviterID: &inviterID, CreatedAt: time.Now().Add(-time.Hour)},
		inviterSummary: &AffiliateSummary{UserID: inviterID, CreatedAt: time.Now().Add(-time.Hour)},
	}
	settings := NewSettingService(&paymentFulfillmentSettingRepoStub{values: map[string]string{
		SettingKeyAffiliateEnabled: "true", SettingKeyAffiliateRebateRate: "10", SettingKeyAffiliateRebateFreezeHours: "0",
	}}, nil)
	svc := NewRedeemService(repo, userRepo, nil, nil, nil, client, nil, NewAffiliateService(affiliateRepo, settings, nil, nil))
	result, err := svc.Redeem(ctx, 42, code.Code)
	require.NoError(t, err)
	require.True(t, result.IsGift)
	require.Equal(t, 1, userRepo.giftCalls)
	require.Equal(t, 0.1, userRepo.gifted)
	require.Empty(t, affiliateRepo.accrueCalls)
	_, err = svc.Redeem(ctx, 42, code.Code)
	require.ErrorIs(t, err, ErrRedeemCodeUsed)
	require.Equal(t, 1, userRepo.giftCalls)
}
