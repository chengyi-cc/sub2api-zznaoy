package service

import (
	"strings"
	"time"
)

type Group struct {
	ID             int64
	Name           string
	Description    string
	Platform       string
	RateMultiplier float64
	IsExclusive    bool
	Status         string
	Hydrated       bool // indicates the group was loaded from a trusted repository source

	SubscriptionType    string
	DailyLimitUSD       *float64
	WeeklyLimitUSD      *float64
	MonthlyLimitUSD     *float64
	DefaultValidityDays int

	ImagePrice1K *float64
	ImagePrice2K *float64
	ImagePrice4K *float64

	ClaudeCodeOnly                  bool
	FallbackGroupID                 *int64
	FallbackGroupIDOnInvalidRequest *int64

	// Model routing configuration.
	ModelRouting        map[string][]int64
	ModelRoutingEnabled bool

	MCPXMLInject         bool
	SupportedModelScopes []string

	SortOrder int

	// OpenAI group-level routing options.
	AllowMessagesDispatch bool
	RequireOAuthOnly      bool
	RequirePrivacySet     bool
	DefaultMappedModel    string
	ForceOpenAIPriority   bool

	CreatedAt time.Time
	UpdatedAt time.Time

	AccountGroups           []AccountGroup
	AccountCount            int64
	ActiveAccountCount      int64
	RateLimitedAccountCount int64
}

func (g *Group) IsActive() bool {
	return g.Status == StatusActive
}

func (g *Group) IsSubscriptionType() bool {
	return g.SubscriptionType == SubscriptionTypeSubscription
}

func (g *Group) IsFreeSubscription() bool {
	return g.IsSubscriptionType() && g.RateMultiplier == 0
}

func (g *Group) HasDailyLimit() bool {
	return g.DailyLimitUSD != nil && *g.DailyLimitUSD > 0
}

func (g *Group) HasWeeklyLimit() bool {
	return g.WeeklyLimitUSD != nil && *g.WeeklyLimitUSD > 0
}

func (g *Group) HasMonthlyLimit() bool {
	return g.MonthlyLimitUSD != nil && *g.MonthlyLimitUSD > 0
}

// GetImagePrice returns the configured image price for a given size.
func (g *Group) GetImagePrice(imageSize string) *float64 {
	switch imageSize {
	case "1K":
		return g.ImagePrice1K
	case "2K":
		return g.ImagePrice2K
	case "4K":
		return g.ImagePrice4K
	default:
		return g.ImagePrice2K
	}
}

// IsGroupContextValid reports whether a group from context has the fields required for routing decisions.
func IsGroupContextValid(group *Group) bool {
	if group == nil {
		return false
	}
	if group.ID <= 0 {
		return false
	}
	if !group.Hydrated {
		return false
	}
	if group.Platform == "" || group.Status == "" {
		return false
	}
	return true
}

// GetRoutingAccountIDs returns the configured account ids for the requested model.
func (g *Group) GetRoutingAccountIDs(requestedModel string) []int64 {
	if !g.ModelRoutingEnabled || len(g.ModelRouting) == 0 || requestedModel == "" {
		return nil
	}

	if accountIDs, ok := g.ModelRouting[requestedModel]; ok && len(accountIDs) > 0 {
		return accountIDs
	}

	for pattern, accountIDs := range g.ModelRouting {
		if matchModelPattern(pattern, requestedModel) && len(accountIDs) > 0 {
			return accountIDs
		}
	}

	return nil
}

// matchModelPattern supports exact matches and suffix '*' prefix matching.
func matchModelPattern(pattern, model string) bool {
	if pattern == model {
		return true
	}

	if strings.HasSuffix(pattern, "*") {
		prefix := strings.TrimSuffix(pattern, "*")
		return strings.HasPrefix(model, prefix)
	}

	return false
}
