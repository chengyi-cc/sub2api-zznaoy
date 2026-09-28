package service

import "context"

type openAIRPMOverflowKey struct{}

func openAIRPMOverflowMode(ctx context.Context) bool {
	enabled, _ := ctx.Value(openAIRPMOverflowKey{}).(bool)
	return enabled
}

func (a *Account) IsOpenAIRPMOverflowEnabled() bool {
	if a == nil || !a.IsOpenAIOAuth() {
		return false
	}
	enabled, _ := a.Extra["openai_rpm_overflow"].(bool)
	return enabled && a.GetBaseRPM() > 0
}

func openAIRPMCanOverflow(ctx context.Context, a *Account) bool {
	return openAIRPMOverflowMode(ctx) && a.IsOpenAIRPMOverflowEnabled()
}

// In overflow mode the regular scheduler still validates model, group,
// credentials, cooldowns and concurrency. Only the local RPM ceiling relaxes.
func leastRPMOverflowAccounts(ctx context.Context, accounts []*Account) []*Account {
	if !openAIRPMOverflowMode(ctx) {
		return accounts
	}
	out := make([]*Account, 0, len(accounts))
	min := int(^uint(0) >> 1)
	for _, a := range accounts {
		if !a.IsOpenAIRPMOverflowEnabled() {
			continue
		}
		state, ok := accountRPMStateFromContext(ctx, a)
		if !ok {
			continue
		}
		if state.Current < min {
			min = state.Current
			out = out[:0]
		}
		if state.Current == min {
			out = append(out, a)
		}
	}
	return out
}
