package service

import (
	"net/http"

	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
)

func (s *OpenAIGatewayService) SetPluginManager(manager *PluginManager) {
	s.pluginManager = manager
}

// doOpenAIUpstream 只在 OpenAI OAuth 能力绑定已启用时把真实请求交给插件。
// 插件返回标准 http.Response，响应解析、错误映射、SSE 和计费仍由现有核心链处理。
func (s *OpenAIGatewayService) doOpenAIUpstream(request *http.Request, proxyURL string, account *Account) (*http.Response, error) {
	if account != nil && account.IsOpenAIOAuthLike() && (s.cfg == nil || !s.cfg.Gateway.DisableCodexZstdRequestBody) {
		request = request.WithContext(WithCodexRequestCompression(request.Context()))
	}
	if s.pluginManager != nil {
		response, handled, err := s.pluginManager.RoundTripOpenAIOAuth(request.Context(), request, proxyURL, account)
		if handled {
			return response, err
		}
	}
	if account.IsTLSFingerprintEnabled() {
		return s.httpUpstream.DoWithTLS(request, proxyURL, account.ID, account.Concurrency, resolveOpenAITransportTLSProfile(s.tlsFPProfileService, account))
	}
	return s.httpUpstream.Do(request, proxyURL, account.ID, account.Concurrency)
}

// doOpenAIAccountTestUpstream 让 OpenAI OAuth 账号测试与真实转发使用同一插件路径。
// API Key 和未命中插件的账号保持各自原有的 HTTPUpstream 行为。
func (s *AccountTestService) doOpenAIAccountTestUpstream(
	request *http.Request,
	proxyURL string,
	account *Account,
	useTLSFallback bool,
) (*http.Response, error) {
	if account != nil && account.IsOpenAIOAuthLike() && (s.cfg == nil || !s.cfg.Gateway.DisableCodexZstdRequestBody) {
		request = request.WithContext(WithCodexRequestCompression(request.Context()))
	}
	if s.pluginManager != nil {
		response, handled, err := s.pluginManager.RoundTripOpenAIOAuth(request.Context(), request, proxyURL, account)
		if handled {
			return response, err
		}
	}
	if useTLSFallback || account.IsTLSFingerprintEnabled() {
		return s.httpUpstream.DoWithTLS(
			request,
			proxyURL,
			account.ID,
			account.Concurrency,
			resolveOpenAITransportTLSProfile(s.tlsFPProfileService, account),
		)
	}
	return s.httpUpstream.Do(request, proxyURL, account.ID, account.Concurrency)
}

func resolveOpenAITransportTLSProfile(profiles *TLSFingerprintProfileService, account *Account) *tlsfingerprint.Profile {
	if profiles != nil {
		return profiles.ResolveTLSProfile(account)
	}
	if account.IsTLSFingerprintEnabled() {
		return &tlsfingerprint.Profile{Name: "Built-in Default (Node.js 24.x)"}
	}
	return nil
}
