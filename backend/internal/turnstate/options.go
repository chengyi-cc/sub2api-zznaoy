package turnstate

import (
	"context"
	"errors"
	"strings"
)

const ProfileKey = "codex_turn_state_profile"
const SourceKey = "codex_turn_state_source"
const ProfileTeam = "team"
const ProfilePro = "pro"
const SourcePurchased = "purchased"
const SourceIPv6 = "ipv6_pool"
const DefaultCountries = "DE,GB,FR,SG,ZA,BR,AE,US,JP,NL,CA,AU,IT,ES,SE,NO,CH,PL,KR,IN"

type Options struct {
	Profile string `json:"profile"`
	Source  string `json:"source"`
}

func (options Options) Normalized() Options {
	if options.Profile != ProfilePro {
		options.Profile = ProfileTeam
	}
	if options.Source != SourceIPv6 {
		options.Source = SourcePurchased
	}
	return options
}

func OptionsFromExtra(extra map[string]any) Options {
	profile, _ := extra[ProfileKey].(string)
	source, _ := extra[SourceKey].(string)
	return (Options{Profile: profile, Source: source}).Normalized()
}

func ValidateOptions(extra map[string]any) error {
	for key, allowed := range map[string][]string{ProfileKey: {ProfileTeam, ProfilePro}, SourceKey: {SourcePurchased, SourceIPv6}} {
		value, exists := extra[key]
		if !exists {
			continue
		}
		text, ok := value.(string)
		if !ok || (text != allowed[0] && text != allowed[1]) {
			return errors.New("invalid automatic turn-state profile or source")
		}
	}
	return nil
}

func acceptedLength(profile string) int {
	if profile == ProfilePro {
		return 292
	}
	return 332
}

type acquisitionOptions struct {
	Options
	Country string
}

type acquisitionKey struct{}

func sampleOptions(ctx context.Context) acquisitionOptions {
	options, _ := ctx.Value(acquisitionKey{}).(acquisitionOptions)
	options.Options = options.Options.Normalized()
	return options
}

func parseCountries(value string) ([]string, error) {
	if strings.TrimSpace(value) == "" {
		value = DefaultCountries
	}
	countries := []string{}
	seen := map[string]bool{}
	for _, country := range strings.Split(strings.ToUpper(value), ",") {
		country = strings.TrimSpace(country)
		if len(country) != 2 || country[0] < 'A' || country[0] > 'Z' || country[1] < 'A' || country[1] > 'Z' {
			return nil, errors.New("proxy countries must be two-letter country codes")
		}
		if !seen[country] {
			countries = append(countries, country)
			seen[country] = true
		}
	}
	if len(countries) > 20 {
		return nil, errors.New("at most 20 proxy countries may be configured")
	}
	return countries, nil
}
