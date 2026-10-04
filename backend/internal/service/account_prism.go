package service

import (
	"github.com/Wei-Shaw/sub2api/internal/config"
	"strings"
)

func accountUsesPrismBrowser(account *Account, cfg *config.Config) bool {
	return accountHasPrismBrowser(account) && cfg != nil && cfg.Gateway.PrismBrowser.Enabled
}

func accountHasPrismBrowser(account *Account) bool {
	if account == nil || !account.IsOpenAIOAuth() || account.IsShadow() || account.IsOpenAIAgentIdentity() || account.IsOpenAIPersonalAccessToken() {
		return false
	}
	for _, key := range []string{openAIAuthModeCredentialKey, openAIAuthModeLegacyCredentialKey} {
		switch strings.ToLower(strings.TrimSpace(account.GetCredential(key))) {
		case "agentidentity", "agent_identity", "personalaccesstoken", "personal_access_token":
			return false
		}
	}
	enabled, _ := account.Extra["openai_prism_browser"].(bool)
	return enabled
}

func prismBrowserResponsesURL(baseURL string) string {
	base := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if base == "" || strings.HasSuffix(base, "/responses") {
		return base
	}
	return base + "/responses"
}

const PrismBrowserModelsKey = "openai_prism_browser_models"

var prismBrowserModels = [...]string{"gpt-6.1-sol", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-luna"}

// PrismBrowserSupportedModels returns the adapter contract, not a statement of
// live account entitlement. Never treat an absent scope as all OpenAI models.
func PrismBrowserSupportedModels() []string {
	return append([]string(nil), prismBrowserModels[:]...)
}

func isPrismBrowserModel(model string) bool {
	for _, supported := range prismBrowserModels {
		if model == supported {
			return true
		}
	}
	return false
}

// IsPrismBrowserEnabledForModel follows the same account-mapping order as BPS.
// Models outside this scope retain their ordinary native/BPS routing.
func (a *Account) IsPrismBrowserEnabledForModel(requestedModel string) bool {
	if !accountHasPrismBrowser(a) {
		return false
	}
	upstream := a.GetMappedModel(strings.TrimSpace(requestedModel))
	return a.isPrismBrowserUpstreamModelEnabled(upstream)
}

func (a *Account) isPrismBrowserUpstreamModelEnabled(upstream string) bool {
	upstream = strings.TrimSpace(upstream)
	if !accountHasPrismBrowser(a) || !isPrismBrowserModel(upstream) {
		return false
	}
	raw, configured := a.Extra[PrismBrowserModelsKey]
	if !configured {
		return true // Legacy enabled accounts inherit only the four known models.
	}
	switch models := raw.(type) {
	case []string:
		for _, model := range models {
			if strings.TrimSpace(model) == upstream {
				return true
			}
		}
	case []any:
		for _, value := range models {
			if model, ok := value.(string); ok && strings.TrimSpace(model) == upstream {
				return true
			}
		}
	}
	return false // Explicit empty or malformed scopes never widen routing.
}
