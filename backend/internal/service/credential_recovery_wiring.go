package service

import "github.com/Wei-Shaw/sub2api/internal/config"

func ProvideOpenAIOAuthReauthService(repo OpenAIOAuthReauthRepository, admin AdminService, accounts AccountRepository,
	oauth *OpenAIOAuthService, encryptor OpenAICredentialEncryptor, cfg *config.Config,
	invalidator TokenCacheInvalidator, blocker AccountRuntimeBlocker, info BuildInfo, settings SettingRepository) *OpenAIOAuthReauthService {
	updater, _ := accounts.(OpenAIOAuthReauthCredentialUpdater)
	svc := NewOpenAIOAuthReauthService(repo, admin, updater, oauth, encryptor, cfg != nil && cfg.Totp.EncryptionKeyConfigured, invalidator, blocker)
	svc.settings = settings
	// Managed rotation is not part of this port. Existing account/saved proxies work.
	svc.acquireMihomoProxy = nil
	svc.configureWorker(cfg, info)
	return svc
}

func ProvideAccountTokenGuardV2Service(repo AccountTokenGuardV2Repository, settings SettingRepository, admin AdminService,
	gateway *OpenAIGatewayService, reauth *OpenAIOAuthReauthService, cfg *config.Config) *AccountTokenGuardV2Service {
	svc := NewAccountTokenGuardV2Service(repo, settings, admin, gateway, reauth)
	svc.Start()
	return svc
}
