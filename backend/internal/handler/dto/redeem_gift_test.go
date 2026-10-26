package dto

import (
	"encoding/json"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/stretchr/testify/require"
)

func TestRedeemGiftDTO_PairingIsAdminOnly(t *testing.T) {
	pairID := "0123456789abcdef0123456789abcdef"
	code := &service.RedeemCode{Code: "own-code", Type: service.RedeemTypeBalance, Value: 0.1, IsGift: true, GiftPairID: &pairID}
	userJSON, err := json.Marshal(RedeemCodeFromService(code))
	require.NoError(t, err)
	require.Contains(t, string(userJSON), `"is_gift":true`)
	require.NotContains(t, string(userJSON), "gift_pair_id")
	require.NotContains(t, string(userJSON), pairID)
	adminJSON, err := json.Marshal(RedeemCodeFromServiceAdmin(code))
	require.NoError(t, err)
	require.Contains(t, string(adminJSON), pairID)
}
