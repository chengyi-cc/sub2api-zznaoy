package service

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/stretchr/testify/require"
)

func TestPrismRuntimeRejectsRemoteAndArbitraryOperations(t *testing.T) {
	for _, endpoint := range []string{"https://127.0.0.1:8320", "http://example.com:8320", "http://localhost:8320", "http://127.0.0.1:8320/status", "http://secret@127.0.0.1:8320", "http://127.0.0.1:8320?key=x"} {
		_, err := prismManagementEndpoint(endpoint, "start")
		require.Error(t, err, endpoint)
	}
	_, err := prismManagementEndpoint("http://127.0.0.1:8320", "../../shell")
	require.Error(t, err)
}

func TestPrismRuntimeControlAndLogSanitization(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "Bearer bridge-secret", r.Header.Get("Authorization"))
		require.Empty(t, r.Header.Get("X-Prism-OAuth-Token"))
		require.Equal(t, http.MethodPost, r.Method)
		require.Equal(t, "/start", r.URL.Path)
		body, _ := io.ReadAll(r.Body)
		require.Empty(t, body)
		_, _ = io.WriteString(w, `{"managed":true,"state":"starting","logs":[{"id":1,"time":"secret","code":"secret"}],"secret":"ignored"}`)
	}))
	defer server.Close()
	result, err := queryPrismRuntime(context.Background(), config.GatewayPrismBrowserConfig{Enabled: true, ManagementURL: server.URL, APIKey: "bridge-secret"}, "start")
	require.NoError(t, err)
	require.True(t, result.GatewayEnabled)
	require.Equal(t, "starting", result.State)
	require.Equal(t, "unknown_event", result.Logs[0].Code)
	require.Empty(t, result.Logs[0].Time)
}

func TestPrismRuntimeDoesNotFollowRedirects(t *testing.T) {
	gotSecret := false
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { gotSecret = true }))
	defer target.Close()
	source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, target.URL, 307) }))
	defer source.Close()
	_, err := queryPrismRuntime(context.Background(), config.GatewayPrismBrowserConfig{ManagementURL: source.URL, APIKey: "bridge-secret"}, "start")
	require.Error(t, err)
	require.False(t, gotSecret)
	require.NotContains(t, err.Error(), "bridge-secret")
}

func TestPrismRuntimeUnmanagedCannotBeControlled(t *testing.T) {
	cfg := config.GatewayPrismBrowserConfig{Enabled: true, APIKey: "external-key"}
	status, err := queryPrismRuntime(context.Background(), cfg, "status")
	require.NoError(t, err)
	require.Equal(t, "unmanaged", status.State)
	require.False(t, status.Managed)
	_, err = queryPrismRuntime(context.Background(), cfg, "stop")
	require.Error(t, err)
}

func TestPrismRuntimeBoundsReplies(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, strings.Repeat("x", 65537)) }))
	defer server.Close()
	_, err := queryPrismRuntime(context.Background(), config.GatewayPrismBrowserConfig{ManagementURL: server.URL, APIKey: "bridge-secret"}, "logs")
	require.Error(t, err)
}
