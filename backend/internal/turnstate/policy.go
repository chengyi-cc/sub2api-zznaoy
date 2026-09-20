package turnstate

import (
	"errors"
	"strings"
	"time"
	"unicode"
)

const DefaultRefreshAfterMinutes = 48
const DefaultIncludedModels = "gpt-6-astra,gpt-5.6-sol,gpt-5.5"

func NormalizeAcquisitionPolicy(config Config) (Config, error) {
	if config.RequireValidState == nil {
		enabled := true
		config.RequireValidState = &enabled
	} else {
		enabled := *config.RequireValidState
		config.RequireValidState = &enabled
	}
	if config.RefreshOnRejection == nil {
		enabled := true
		config.RefreshOnRejection = &enabled
	} else {
		enabled := *config.RefreshOnRejection
		config.RefreshOnRejection = &enabled
	}
	if config.RefreshAfterMinutes == 0 {
		config.RefreshAfterMinutes = DefaultRefreshAfterMinutes
	}
	if config.RefreshAfterMinutes < 1 || config.RefreshAfterMinutes > 59 {
		return config, errors.New("刷新时间须为签发后1–59分钟")
	}
	if config.IncludedModels == nil {
		config.IncludedModels = strings.Split(DefaultIncludedModels, ",")
	}
	if len(config.IncludedModels) > 100 {
		return config, errors.New("最多指定100个采集模型")
	}
	models := []string{}
	seen := map[string]bool{}
	for _, model := range config.IncludedModels {
		model = strings.ToLower(strings.TrimSpace(model))
		if model == "" || len(model) > 256 || strings.ContainsAny(model, ",*") || strings.ContainsFunc(model, func(character rune) bool { return unicode.IsSpace(character) || unicode.IsControl(character) }) {
			return config, errors.New("采集模型须为完整模型名称，不支持空白或通配符")
		}
		if !seen[model] {
			models = append(models, model)
			seen[model] = true
		}
	}
	config.IncludedModels = models
	return config, nil
}

func (manager *Manager) ModelExcluded(model string) bool {
	if manager == nil {
		return false
	}
	return !modelIncluded(manager.config, model)
}

func modelIncluded(config Config, model string) bool {
	for _, included := range config.IncludedModels {
		if strings.EqualFold(strings.TrimSpace(model), included) {
			return true
		}
	}
	return false
}

func (manager *Manager) refreshBefore() time.Duration {
	return lifetime - time.Duration(manager.config.RefreshAfterMinutes)*time.Minute
}
