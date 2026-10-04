package repository

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/stretchr/testify/require"
)

func (s *HTTPUpstreamSuite) TestAstraProbeFreshHTTP1DespiteHTTP2Setting() {
	s.cfg.Gateway.OpenAIHTTP2 = config.GatewayOpenAIHTTP2Config{Enabled: true}
	svc := s.newService()
	ordinary, err := svc.getClientEntry("", 1, 1, service.HTTPUpstreamProfileOpenAI, false, false)
	require.NoError(s.T(), err)
	probe, err := svc.getClientEntry("", 1, 1, service.HTTPUpstreamProfileOpenAIHarvest, false, false)
	require.NoError(s.T(), err)
	require.NotSame(s.T(), ordinary, probe)
	require.Equal(s.T(), upstreamProtocolModeOpenAIH2, ordinary.protocolMode)

	type observed struct {
		protocol int
		remote   string
	}
	observations := make(chan observed, 2)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		observations <- observed{r.ProtoMajor, r.RemoteAddr}
		_, _ = io.WriteString(w, "OK")
	}))
	server.EnableHTTP2 = true
	server.StartTLS()
	s.T().Cleanup(server.Close)
	transport := probe.client.Transport.(*http.Transport)
	transport.TLSClientConfig = server.Client().Transport.(*http.Transport).TLSClientConfig.Clone()
	s.T().Cleanup(probe.client.CloseIdleConnections)
	for range 2 {
		// Do not set req.Close: the profile itself must enforce fresh connections.
		response, err := probe.client.Get(server.URL)
		require.NoError(s.T(), err)
		_, err = io.Copy(io.Discard, response.Body)
		require.NoError(s.T(), err)
		require.NoError(s.T(), response.Body.Close())
	}
	first, second := <-observations, <-observations
	require.Equal(s.T(), 1, first.protocol)
	require.Equal(s.T(), 1, second.protocol)
	require.NotEqual(s.T(), first.remote, second.remote, "each shot must establish a new connection")
}

func (s *HTTPUpstreamSuite) TestAstraProbeFingerprintPoolIsolated() {
	svc := s.newService()
	profile := &tlsfingerprint.Profile{Name: "test", ALPNProtocols: []string{"h2", "http/1.1"}}
	ordinary, err := svc.getClientEntryWithTLS("", 1, 1, profile, service.HTTPUpstreamProfileOpenAI, false, false)
	require.NoError(s.T(), err)
	probe, err := svc.getClientEntryWithTLS("", 1, 1, profile, service.HTTPUpstreamProfileOpenAIHarvest, false, false)
	require.NoError(s.T(), err)
	require.NotSame(s.T(), ordinary, probe)
	require.False(s.T(), ordinary.client.Transport.(*http.Transport).DisableKeepAlives)
	require.True(s.T(), probe.client.Transport.(*http.Transport).DisableKeepAlives)
	require.False(s.T(), probe.client.Transport.(*http.Transport).ForceAttemptHTTP2)
	require.Equal(s.T(), []string{"h2", "http/1.1"}, profile.ALPNProtocols, "do not mutate the account fingerprint")
}

func TestAstraProbeProfileSurvivesContext(t *testing.T) {
	ctx := service.WithHTTPUpstreamProfile(t.Context(), service.HTTPUpstreamProfileOpenAIHarvest)
	require.Equal(t, service.HTTPUpstreamProfileOpenAIHarvest, service.HTTPUpstreamProfileFromContext(ctx))
}
