package turnstate

import (
	"errors"
	"strings"
	"time"
	"unicode"
)

const DefaultRefreshAfterMinutes = 48
const DefaultExcludedModels = "codex-auto-review,gpt-5.6-terra,gpt-5.4"

func NormalizeAcquisitionPolicy(config Config) (Config, error) {
	if config.RefreshAfterMinutes == 0 {
		config.RefreshAfterMinutes = DefaultRefreshAfterMinutes
	}
	if config.RefreshAfterMinutes < 1 || config.RefreshAfterMinutes > 59 {
		return config, errors.New("刷新时间须为签发后1–59分钟")
	}
	if config.ExcludedModels == nil {
		config.ExcludedModels = strings.Split(DefaultExcludedModels, ",")
	}
	if len(config.ExcludedModels) > 100 {
		return config, errors.New("最多排除100个采集模型")
	}
	models := []string{}
	seen := map[string]bool{}
	for _, model := range config.ExcludedModels {
		model = strings.ToLower(strings.TrimSpace(model))
		if model == "" || len(model) > 256 || strings.ContainsAny(model, ",*") || strings.ContainsFunc(model, func(character rune) bool { return unicode.IsSpace(character) || unicode.IsControl(character) }) {
			return config, errors.New("排除模型须为完整模型名称，不支持空白或通配符")
		}
		if !seen[model] {
			models = append(models, model)
			seen[model] = true
		}
	}
	config.ExcludedModels = models
	return config, nil
}

func (manager *Manager) ModelExcluded(model string) bool {
	if manager == nil {
		return false
	}
	for _, excluded := range manager.config.ExcludedModels {
		if strings.EqualFold(strings.TrimSpace(model), excluded) {
			return true
		}
	}
	return false
}

func (manager *Manager) refreshBefore() time.Duration {
	return lifetime - time.Duration(manager.config.RefreshAfterMinutes)*time.Minute
}
