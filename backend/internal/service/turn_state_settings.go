package service

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"sync"
	"time"

	infraerrors "github.com/Wei-Shaw/sub2api/internal/pkg/errors"
	"github.com/Wei-Shaw/sub2api/internal/turnstate"
)

const turnStateSettingsKey = "codex_turn_state_acquisition_encrypted_v1"

type TurnStateSettingsView struct {
	Revision                string   `json:"revision"`
	PurchasedEnabled        bool     `json:"purchased_enabled"`
	ProxyHost               string   `json:"proxy_host"`
	ProxyUsername           string   `json:"proxy_username"`
	ProxyPassword           string   `json:"proxy_password,omitempty"`
	ProxyPasswordConfigured bool     `json:"proxy_password_configured"`
	ProxyUpstream           string   `json:"proxy_upstream,omitempty"`
	ProxyUpstreamConfigured bool     `json:"proxy_upstream_configured"`
	ClearProxyUpstream      bool     `json:"clear_proxy_upstream,omitempty"`
	Countries               string   `json:"countries"`
	IPv6Enabled             bool     `json:"ipv6_enabled"`
	PoolURL                 string   `json:"pool_url"`
	PoolToken               string   `json:"pool_token,omitempty"`
	PoolTokenConfigured     bool     `json:"pool_token_configured"`
	PoolCA                  string   `json:"pool_ca"`
	Attempts                int      `json:"attempts"`
	Concurrency             int      `json:"concurrency"`
	RefreshAfterMinutes     int      `json:"refresh_after_minutes"`
	IncludedModels          []string `json:"included_models"`
	RefreshOnRejection      *bool    `json:"refresh_on_rejection"`
	RequireValidState       *bool    `json:"require_valid_state"`
}

type turnStateStoredSettings struct {
	PurchasedEnabled bool
	IPv6Enabled      bool
	Config           turnstate.Config
}

func (settings turnStateStoredSettings) effective() turnstate.Config {
	config := settings.Config
	if !settings.PurchasedEnabled {
		config.ProxyHost, config.ProxyUsername, config.ProxyPassword, config.ProxyUpstream = "", "", "", ""
	}
	if !settings.IPv6Enabled {
		config.URL, config.Token, config.CAFile, config.CAPEM = "", "", "", ""
	}
	return config
}

type turnStateSettings struct {
	mu       sync.Mutex
	repo     SettingRepository
	cipher   SecretEncryptor
	runtime  *turnstate.Runtime
	fallback turnStateStoredSettings
	revision string
	cancel   context.CancelFunc
	done     chan struct{}
}

func (gateway *OpenAIGatewayService) initializeTurnStateSettings(fallback turnstate.Config) {
	settings := &turnStateSettings{runtime: gateway.turnStateAuto, done: make(chan struct{}), fallback: turnStateStoredSettings{
		PurchasedEnabled: fallback.ProxyHost != "" || fallback.ProxyUsername != "" || fallback.ProxyPassword != "",
		IPv6Enabled:      fallback.URL != "" || fallback.Token != "", Config: fallback,
	}}
	if gateway.settingService != nil {
		settings.repo = gateway.settingService.settingRepo
	}
	if gateway.cfg != nil && gateway.cfg.JWT.Secret != "" {
		settings.cipher = &liveAttestationAES{key: sha256.Sum256([]byte("sub2api/turn-state-settings/v1\x00" + gateway.cfg.JWT.Secret))}
	}
	ctx, cancel := context.WithCancel(context.Background())
	settings.cancel = cancel
	gateway.turnStateSettings = settings
	settings.sync(ctx)
	go func() {
		defer close(settings.done)
		ticker := time.NewTicker(15 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				settings.sync(ctx)
			}
		}
	}()
}

func (settings *turnStateSettings) close() {
	settings.cancel()
	<-settings.done
}

func (settings *turnStateSettings) load(ctx context.Context) (turnStateStoredSettings, string, error) {
	if settings.repo == nil {
		return settings.fallback, "environment", nil
	}
	value, err := settings.repo.GetValue(ctx, turnStateSettingsKey)
	if errors.Is(err, ErrSettingNotFound) || (err == nil && value == "") {
		return settings.fallback, "environment", nil
	}
	if err != nil {
		return turnStateStoredSettings{}, "", errors.New("无法读取采集配置，请稍后重试")
	}
	if settings.cipher == nil {
		return turnStateStoredSettings{}, "", errors.New("采集配置加密服务不可用")
	}
	decoded, err := settings.cipher.Decrypt(value)
	if err != nil {
		return turnStateStoredSettings{}, "", errors.New("无法解密采集配置，请检查服务器加密密钥是否变化")
	}
	var stored turnStateStoredSettings
	if json.Unmarshal([]byte(decoded), &stored) != nil {
		return stored, "", errors.New("采集配置内容无效")
	}
	digest := sha256.Sum256([]byte(value))
	return stored, hex.EncodeToString(digest[:]), nil
}

func (settings *turnStateSettings) sync(parent context.Context) {
	settings.mu.Lock()
	defer settings.mu.Unlock()
	ctx, cancel := context.WithTimeout(parent, 5*time.Second)
	defer cancel()
	stored, revision, err := settings.load(ctx)
	if err == nil && revision != settings.revision {
		err = settings.runtime.Update(stored.effective(), nil)
		if err == nil {
			settings.revision = revision
		}
	}
	if err != nil {
		reportTurnStateConfigurationError(err)
	}
}

func turnStateSettingsView(stored turnStateStoredSettings, revision string) TurnStateSettingsView {
	config := stored.Config
	if normalized, err := turnstate.NormalizeAcquisitionPolicy(config); err == nil {
		config = normalized
	}
	countries := strings.Join(config.Countries, ",")
	if strings.Trim(countries, ", ") == "" {
		countries = turnstate.DefaultCountries
	}
	certificate := config.CAPEM
	if certificate == "" && config.CAFile != "" {
		if contents, err := os.ReadFile(config.CAFile); err == nil {
			certificate = string(contents)
		}
	}
	return TurnStateSettingsView{Revision: revision, PurchasedEnabled: stored.PurchasedEnabled,
		ProxyHost: config.ProxyHost, ProxyUsername: config.ProxyUsername, ProxyPasswordConfigured: config.ProxyPassword != "",
		ProxyUpstreamConfigured: config.ProxyUpstream != "", Countries: countries, IPv6Enabled: stored.IPv6Enabled,
		PoolURL: config.URL, PoolTokenConfigured: config.Token != "", PoolCA: certificate, Attempts: config.Attempts, Concurrency: config.Concurrency,
		RefreshAfterMinutes: config.RefreshAfterMinutes, IncludedModels: config.IncludedModels, RefreshOnRejection: config.RefreshOnRejection, RequireValidState: config.RequireValidState}
}

func (gateway *OpenAIGatewayService) GetTurnStateSettings(ctx context.Context) (TurnStateSettingsView, error) {
	if gateway == nil || gateway.turnStateSettings == nil {
		return TurnStateSettingsView{}, errors.New("采集配置服务不可用")
	}
	settings := gateway.turnStateSettings
	settings.mu.Lock()
	defer settings.mu.Unlock()
	stored, revision, err := settings.load(ctx)
	if err != nil {
		return TurnStateSettingsView{}, err
	}
	return turnStateSettingsView(stored, revision), nil
}

func (gateway *OpenAIGatewayService) SaveTurnStateSettings(ctx context.Context, request TurnStateSettingsView) (TurnStateSettingsView, error) {
	if gateway == nil || gateway.turnStateSettings == nil {
		return TurnStateSettingsView{}, errors.New("采集配置服务不可用")
	}
	settings := gateway.turnStateSettings
	settings.mu.Lock()
	defer settings.mu.Unlock()
	if settings.repo == nil || settings.cipher == nil {
		return TurnStateSettingsView{}, errors.New("采集配置存储或加密服务不可用")
	}
	stored, revision, err := settings.load(ctx)
	if err != nil {
		return TurnStateSettingsView{}, err
	}
	if revision != request.Revision {
		return TurnStateSettingsView{}, infraerrors.Conflict("TURN_STATE_CONFIG_CHANGED", "配置已被其他管理员修改，请关闭后重新打开配置")
	}
	if request.Attempts < 1 || request.Attempts > 30 || request.Concurrency < 1 || request.Concurrency > 16 {
		return TurnStateSettingsView{}, infraerrors.BadRequest("TURN_STATE_CONFIG_INVALID", "每轮尝试次数须为1–30，并发任务数须为1–16")
	}
	stored.PurchasedEnabled, stored.IPv6Enabled = request.PurchasedEnabled, request.IPv6Enabled
	config := &stored.Config
	config.ProxyHost, config.ProxyUsername = strings.TrimSpace(request.ProxyHost), strings.TrimSpace(request.ProxyUsername)
	config.URL = strings.TrimRight(strings.TrimSpace(request.PoolURL), "/")
	config.Countries = strings.Split(strings.ToUpper(strings.TrimSpace(request.Countries)), ",")
	config.Attempts, config.Concurrency = request.Attempts, request.Concurrency
	if request.RefreshAfterMinutes != 0 {
		config.RefreshAfterMinutes = request.RefreshAfterMinutes
	}
	if request.IncludedModels != nil {
		config.IncludedModels = request.IncludedModels
	}
	if request.RefreshOnRejection != nil {
		config.RefreshOnRejection = request.RefreshOnRejection
	}
	if request.RequireValidState != nil {
		config.RequireValidState = request.RequireValidState
	}
	normalized, err := turnstate.NormalizeAcquisitionPolicy(*config)
	if err != nil {
		return TurnStateSettingsView{}, infraerrors.BadRequest("TURN_STATE_CONFIG_INVALID", err.Error())
	}
	*config = normalized
	config.CAFile, config.CAPEM = "", strings.TrimSpace(request.PoolCA)
	if request.ProxyPassword != "" {
		config.ProxyPassword = request.ProxyPassword
	}
	if request.PoolToken != "" {
		config.Token = strings.TrimSpace(request.PoolToken)
	}
	if request.ClearProxyUpstream {
		config.ProxyUpstream = ""
	} else if request.ProxyUpstream != "" {
		config.ProxyUpstream = strings.TrimSpace(request.ProxyUpstream)
	}
	if stored.PurchasedEnabled && (config.ProxyHost == "" || config.ProxyUsername == "" || config.ProxyPassword == "") {
		return TurnStateSettingsView{}, infraerrors.BadRequest("TURN_STATE_CONFIG_INVALID", "启用IPv4需要填写代理地址、用户名模板和密码")
	}
	if stored.IPv6Enabled && (config.URL == "" || config.Token == "") {
		return TurnStateSettingsView{}, infraerrors.BadRequest("TURN_STATE_CONFIG_INVALID", "启用IPv6需要填写池地址和管理密钥")
	}
	payload, err := json.Marshal(stored)
	if err != nil {
		return TurnStateSettingsView{}, errors.New("无法编码采集配置")
	}
	encrypted, err := settings.cipher.Encrypt(string(payload))
	if err != nil {
		return TurnStateSettingsView{}, errors.New("无法加密采集配置")
	}
	var persistenceError error
	err = settings.runtime.Update(stored.effective(), func() error {
		persistenceError = settings.repo.Set(ctx, turnStateSettingsKey, encrypted)
		if persistenceError != nil {
			return errors.New("无法保存采集配置，原配置保持不变")
		}
		return nil
	})
	if err != nil {
		if persistenceError != nil {
			return TurnStateSettingsView{}, errors.New("无法保存采集配置，原配置保持不变")
		}
		return TurnStateSettingsView{}, infraerrors.BadRequest("TURN_STATE_CONFIG_INVALID", err.Error())
	}
	digest := sha256.Sum256([]byte(encrypted))
	settings.revision = hex.EncodeToString(digest[:])
	return turnStateSettingsView(stored, settings.revision), nil
}
