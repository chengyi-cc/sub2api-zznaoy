package service

import "context"

// IsOpenAIGPT61SolLongContextExemptionEnabled requires an explicit opt-in.
// Read storage for each eligible billing operation so saves affect every instance
// without stale local caches. Missing, malformed or unreadable settings never waive fees.
func (s *SettingService) IsOpenAIGPT61SolLongContextExemptionEnabled(ctx context.Context) bool {
	if s == nil || s.settingRepo == nil {
		return false
	}
	dbCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), gatewayForwardingDBTimeout)
	defer cancel()
	value, err := s.settingRepo.GetValue(dbCtx, SettingKeyOpenAIGPT61SolLongContextExemptionEnabled)
	return err == nil && value == "true"
}
