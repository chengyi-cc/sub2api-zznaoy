package service

import (
	"context"
	coderws "github.com/coder/websocket"
	"slices"
)

func (s *AstraBorrowService) RejectExcelTarget(ctx context.Context, id int64) error {
	v, err := s.Settings(ctx)
	if err != nil {
		return &AstraBorrowRequestError{Code: err.Error()}
	}
	if v.Enabled && slices.Contains(v.TargetAccountIDs, id) {
		return &AstraBorrowRequestError{Code: "astra_disable_excel_first"}
	}
	return nil
}

func (s *AccountTestService) AstraBorrow() *AstraBorrowService {
	if s == nil || s.openaiGatewayService == nil {
		return nil
	}
	return s.openaiGatewayService.astraBorrow
}

// Check every turn, including a connection opened before borrowing was enabled.
func (s *OpenAIGatewayService) withAstraBorrowWSGuard(ctx context.Context, account *Account, hooks *OpenAIWSIngressHooks) *OpenAIWSIngressHooks {
	if s.astraBorrow == nil || account == nil || !account.IsOpenAIOAuth() {
		return hooks
	}
	guarded := &OpenAIWSIngressHooks{}
	if hooks != nil {
		*guarded = *hooks
	}
	previous := guarded.BeforeRequest
	guarded.BeforeRequest = func(turn int, payload []byte, model string) error {
		if err := s.astraBorrow.RejectWebSocket(ctx, account.ID); err != nil {
			return NewOpenAIWSClientCloseError(coderws.StatusPolicyViolation, err.Error(), err)
		}
		if previous != nil {
			return previous(turn, payload, model)
		}
		return nil
	}
	return guarded
}
