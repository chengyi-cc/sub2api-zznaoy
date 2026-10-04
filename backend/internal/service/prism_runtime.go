package service

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
)

type PrismRuntimeLog struct {
	ID   int64  `json:"id"`
	Time string `json:"time"`
	Code string `json:"code"`
}

type PrismRuntimeStatus struct {
	Managed        bool              `json:"managed"`
	GatewayEnabled bool              `json:"gateway_enabled"`
	State          string            `json:"state"`
	Healthy        bool              `json:"healthy"`
	DesiredEnabled bool              `json:"desired_enabled"`
	Logs           []PrismRuntimeLog `json:"logs,omitempty"`
}

// Admin operations use only a configured numeric loopback endpoint and a fixed
// operation allowlist. Never forward account credentials, client headers or bodies.
func prismManagementEndpoint(base, operation string) (string, error) {
	u, err := url.Parse(strings.TrimSpace(base))
	if err != nil || u.Scheme != "http" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.Opaque != "" || (u.Path != "" && u.Path != "/") {
		return "", errors.New("invalid Prism management endpoint")
	}
	ip := net.ParseIP(u.Hostname())
	port, err := strconv.Atoi(u.Port())
	if err != nil || port < 1 || port > 65535 || ip == nil || (!ip.Equal(net.ParseIP("127.0.0.1")) && !ip.Equal(net.IPv6loopback)) {
		return "", errors.New("invalid Prism management endpoint")
	}
	switch operation {
	case "status", "logs", "start", "stop", "restart", "check":
	default:
		return "", errors.New("invalid Prism service operation")
	}
	u.Path = "/" + operation
	return u.String(), nil
}

func (s *AccountTestService) PrismRuntime(ctx context.Context, operation string) (PrismRuntimeStatus, error) {
	var cfg config.GatewayPrismBrowserConfig
	if s != nil && s.openaiGatewayService != nil && s.openaiGatewayService.cfg != nil {
		cfg = s.openaiGatewayService.cfg.Gateway.PrismBrowser
	}
	return queryPrismRuntime(ctx, cfg, operation)
}

func queryPrismRuntime(ctx context.Context, cfg config.GatewayPrismBrowserConfig, operation string) (PrismRuntimeStatus, error) {
	status := PrismRuntimeStatus{State: "not_installed", GatewayEnabled: cfg.Enabled}
	if cfg.ManagementURL == "" {
		if cfg.Enabled && cfg.APIKey != "" {
			status.State = "unmanaged"
		}
		if operation == "status" {
			return status, nil
		}
		return status, errors.New("Prism service management is not installed")
	}
	endpoint, err := prismManagementEndpoint(cfg.ManagementURL, operation)
	if err != nil || strings.TrimSpace(cfg.APIKey) == "" {
		status.State = "misconfigured"
		if operation == "status" {
			return status, nil
		}
		return status, errors.New("Prism service management is misconfigured")
	}
	method := http.MethodPost
	if operation == "status" || operation == "logs" {
		method = http.MethodGet
	}
	req, err := http.NewRequestWithContext(ctx, method, endpoint, nil)
	if err != nil {
		return status, errors.New("Prism service request failed")
	}
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	transport := &http.Transport{Proxy: nil, DisableKeepAlives: true}
	defer transport.CloseIdleConnections()
	timeout := 15 * time.Second
	if method == http.MethodGet {
		timeout = 3 * time.Second
	}
	client := &http.Client{Transport: transport, Timeout: timeout, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	resp, err := client.Do(req)
	if err != nil {
		status.State = "unreachable"
		if operation == "status" {
			return status, nil
		}
		return status, errors.New("Prism service manager is unreachable")
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 65537))
	if resp.StatusCode != http.StatusOK || err != nil || len(data) > 65536 || json.Unmarshal(data, &status) != nil {
		return PrismRuntimeStatus{State: "error", GatewayEnabled: cfg.Enabled}, errors.New("Prism service manager returned an invalid response")
	}
	status.GatewayEnabled = cfg.Enabled
	switch status.State {
	case "stopped", "starting", "running", "error":
	default:
		return PrismRuntimeStatus{}, errors.New("Prism service manager returned an invalid state")
	}
	if len(status.Logs) > 200 {
		status.Logs = status.Logs[len(status.Logs)-200:]
	}
	for i := range status.Logs {
		entry := &status.Logs[i]
		switch entry.Code {
		case "manager_ready", "port_in_use", "service_starting", "service_start_failed", "service_stopped", "service_exited", "service_ready", "health_failed", "check_ok", "check_failed", "control_failed", "prism_adapter_error", "prism_worker_close_error", "browser_missing", "sandbox_missing", "non_root_required", "adapter_configuration_invalid", "dependencies_missing":
		default:
			entry.Code = "unknown_event"
		}
		if _, err := time.Parse(time.RFC3339Nano, entry.Time); err != nil {
			entry.Time = ""
		}
	}
	return status, nil
}
