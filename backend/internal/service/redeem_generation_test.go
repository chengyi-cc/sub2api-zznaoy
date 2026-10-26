//go:build unit

package service

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"

	infraerrors "github.com/Wei-Shaw/sub2api/internal/pkg/errors"
	"github.com/stretchr/testify/require"
)

type giftGenerationRepo struct {
	redeemRepoStub
	codes []RedeemCode
	calls int
	err   error
}

func (r *giftGenerationRepo) CreateBatch(_ context.Context, codes []RedeemCode) error {
	r.calls++
	r.codes = append([]RedeemCode(nil), codes...)
	return r.err
}

func TestGenerateRedeemCodes_GiftPairs(t *testing.T) {
	for _, tc := range []struct {
		name        string
		value, gift float64
		count       int
	}{
		{"ten", 10, 0.1, 2}, {"smallest", 0.01, 0.0001, 1},
		{"decimal", 29.99, 0.2999, 3}, {"maximum batch", 10, 0.1, 500},
	} {
		t.Run(tc.name, func(t *testing.T) {
			repo := &giftGenerationRepo{}
			svc := &adminServiceImpl{redeemCodeRepo: repo}
			expiry := time.Now().Add(time.Hour)
			codes, err := svc.GenerateRedeemCodes(context.Background(), &GenerateRedeemCodesInput{
				Count: tc.count, Type: RedeemTypeBalance, Value: tc.value, IncludeGift: true, ExpiresAt: &expiry,
			})
			require.NoError(t, err)
			require.Equal(t, 1, repo.calls)
			require.Len(t, codes, tc.count*2)
			secrets := map[string]bool{}
			pairs := map[string]bool{}
			for i := 0; i < len(codes); i += 2 {
				paid, gift := codes[i], codes[i+1]
				require.False(t, paid.IsGift)
				require.True(t, gift.IsGift)
				require.Equal(t, tc.value, paid.Value)
				require.Equal(t, tc.gift, gift.Value)
				require.Equal(t, RedeemTypeBalance, gift.Type)
				require.Equal(t, StatusUnused, gift.Status)
				require.Nil(t, gift.UsedBy)
				require.Nil(t, gift.UsedAt)
				require.Nil(t, gift.GroupID)
				require.Equal(t, &expiry, paid.ExpiresAt)
				require.Equal(t, &expiry, gift.ExpiresAt)
				require.NotNil(t, paid.GiftPairID)
				require.Equal(t, paid.GiftPairID, gift.GiftPairID)
				require.Len(t, *paid.GiftPairID, 32)
				require.False(t, pairs[*paid.GiftPairID])
				pairs[*paid.GiftPairID] = true
				for _, c := range []RedeemCode{paid, gift} {
					require.Len(t, c.Code, 32)
					require.NotEqual(t, *paid.GiftPairID, c.Code)
					require.False(t, secrets[c.Code])
					secrets[c.Code] = true
				}
			}
		})
	}
}

func TestGenerateRedeemCodes_RejectsUnsafeGiftInputsBeforeWrite(t *testing.T) {
	for _, tc := range []struct {
		name  string
		input *GenerateRedeemCodesInput
	}{
		{"nil", nil},
		{"zero count", &GenerateRedeemCodesInput{Count: 0, Type: RedeemTypeBalance, Value: 10, IncludeGift: true}},
		{"excessive count", &GenerateRedeemCodesInput{Count: 501, Type: RedeemTypeBalance, Value: 10, IncludeGift: true}},
		{"negative", &GenerateRedeemCodesInput{Count: 1, Type: RedeemTypeBalance, Value: -10, IncludeGift: true}},
		{"zero", &GenerateRedeemCodesInput{Count: 1, Type: RedeemTypeBalance, Value: 0, IncludeGift: true}},
		{"nan", &GenerateRedeemCodesInput{Count: 1, Type: RedeemTypeBalance, Value: math.NaN(), IncludeGift: true}},
		{"infinity", &GenerateRedeemCodesInput{Count: 1, Type: RedeemTypeBalance, Value: math.Inf(1), IncludeGift: true}},
		{"overflow", &GenerateRedeemCodesInput{Count: 1, Type: RedeemTypeBalance, Value: 1e12, IncludeGift: true}},
		{"sub-cent", &GenerateRedeemCodesInput{Count: 1, Type: RedeemTypeBalance, Value: 0.001, IncludeGift: true}},
		{"excess precision", &GenerateRedeemCodesInput{Count: 1, Type: RedeemTypeBalance, Value: 1.000000001}},
		{"wrong type", &GenerateRedeemCodesInput{Count: 1, Type: RedeemTypeConcurrency, Value: 10, IncludeGift: true}},
		{"subscription", &GenerateRedeemCodesInput{Count: 1, Type: RedeemTypeSubscription, Value: 10, IncludeGift: true}},
		{"invitation", &GenerateRedeemCodesInput{Count: 1, Type: RedeemTypeInvitation, IncludeGift: true}},
		{"unknown type", &GenerateRedeemCodesInput{Count: 1, Type: "gift", Value: 10, IncludeGift: true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			repo := &giftGenerationRepo{}
			codes, err := (&adminServiceImpl{redeemCodeRepo: repo}).GenerateRedeemCodes(context.Background(), tc.input)
			require.Nil(t, codes)
			require.True(t, infraerrors.IsBadRequest(err), "expected bad request: %v", err)
			require.Zero(t, repo.calls)
		})
	}
	repo := &giftGenerationRepo{}
	past := time.Now().Add(-time.Second)
	_, err := (&adminServiceImpl{redeemCodeRepo: repo}).GenerateRedeemCodes(context.Background(), &GenerateRedeemCodesInput{
		Count: 1, Type: RedeemTypeBalance, Value: 10, IncludeGift: true, ExpiresAt: &past,
	})
	require.ErrorIs(t, err, ErrRedeemCodeExpired)
	require.Zero(t, repo.calls)
}

func TestGenerateRedeemCodes_NoGiftAndBatchError(t *testing.T) {
	repo := &giftGenerationRepo{}
	svc := &adminServiceImpl{redeemCodeRepo: repo}
	codes, err := svc.GenerateRedeemCodes(context.Background(), &GenerateRedeemCodesInput{Count: 2, Type: RedeemTypeBalance, Value: 10})
	require.NoError(t, err)
	require.Len(t, codes, 2)
	for _, code := range codes {
		require.False(t, code.IsGift)
		require.Nil(t, code.GiftPairID)
	}
	failure := errors.New("database unavailable")
	repo.err = failure
	codes, err = svc.GenerateRedeemCodes(context.Background(), &GenerateRedeemCodesInput{Count: 1, Type: RedeemTypeBalance, Value: 10, IncludeGift: true})
	require.Nil(t, codes)
	require.ErrorIs(t, err, failure)
}
