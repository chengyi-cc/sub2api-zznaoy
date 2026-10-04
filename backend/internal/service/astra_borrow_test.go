package service

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type astraTestStore struct {
	mu   sync.Mutex
	raw  string
	rows []AstraBorrowHistory
	err  error
}

func (s *astraTestStore) GetValue(context.Context, string) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.raw, s.err
}
func (s *astraTestStore) Set(_ context.Context, _, value string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return s.err
	}
	s.raw = value
	return nil
}
func (s *astraTestStore) AppendAstraBorrowHistory(_ context.Context, row AstraBorrowHistory) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.rows = append(s.rows, row)
	return nil
}
func (s *astraTestStore) ListAstraBorrowHistory(context.Context, int64, int) ([]AstraBorrowHistory, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]AstraBorrowHistory{}, s.rows...), nil
}
func (s *astraTestStore) config(v AstraBorrowSettings) {
	b, _ := json.Marshal(v)
	_ = s.Set(context.Background(), "", string(b))
}

type astraTestAccounts struct {
	AccountRepository
	mu    sync.Mutex
	items map[int64]*Account
}

func (s *astraTestAccounts) GetByID(_ context.Context, id int64) (*Account, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	a := s.items[id]
	if a == nil {
		return nil, ErrAccountNotFound
	}
	data, _ := json.Marshal(a)
	var copy Account
	_ = json.Unmarshal(data, &copy)
	return &copy, nil
}

type astraTestTransport struct {
	fn func(*http.Request, string, int64) (*http.Response, error)
}

func (s *astraTestTransport) Do(r *http.Request, proxy string, id int64, _ int) (*http.Response, error) {
	return s.fn(r, proxy, id)
}
func (s *astraTestTransport) DoWithTLS(r *http.Request, proxy string, id int64, _ int, _ *tlsfingerprint.Profile) (*http.Response, error) {
	return s.fn(r, proxy, id)
}

const astraComplete = "data: {\"type\":\"response.completed\",\"response\":{\"model\":\"gpt-6-astra\",\"status\":\"completed\",\"output\":[{\"content\":[{\"type\":\"output_text\",\"text\":\"OK\"}]}]}}\n\n"

func astraTestResponse(cookie, state, body string) *http.Response {
	h := http.Header{"Content-Type": {"text/event-stream"}}
	if cookie != "" {
		h.Set("Set-Cookie", cookie)
	}
	if state != "" {
		h.Set("x-codex-turn-state", state)
	}
	return &http.Response{StatusCode: 200, Header: h, Body: io.NopCloser(strings.NewReader(body))}
}

func astraFixture(t *testing.T) (*AstraBorrowService, *astraTestStore, *astraTestAccounts, *astraTestTransport, *Account) {
	t.Helper()
	store := &astraTestStore{}
	v := defaultAstraBorrowSettings()
	v.Enabled = true
	v.SourceAccountIDs = []int64{1}
	v.TargetAccountIDs = []int64{2}
	v.Revision = "r1"
	store.config(v)
	accounts := &astraTestAccounts{items: map[int64]*Account{}}
	for _, id := range []int64{1, 2, 3} {
		accounts.items[id] = &Account{ID: id, Platform: PlatformOpenAI, Type: AccountTypeOAuth, Status: "active", Schedulable: true, Concurrency: 1, Credentials: map[string]any{"access_token": map[int64]string{1: "source-secret", 2: "target-secret", 3: "other-secret"}[id]}}
	}
	transport := &astraTestTransport{}
	transport.fn = func(req *http.Request, _ string, id int64) (*http.Response, error) {
		require.Nil(t, req.GetBody, "transport must not rewind sent requests")
		require.True(t, HTTPUpstreamRedirectsDisabled(req.Context()))
		if id == 1 {
			return astraTestResponse("__oailb=route-secret; Secure; Path=/; Max-Age=240", "source-ticket", astraComplete), nil
		}
		return astraTestResponse("", "target-ticket", astraComplete), nil
	}
	g := &OpenAIGatewayService{accountRepo: accounts, httpUpstream: transport}
	s := NewAstraBorrowService(g, store)
	g.astraBorrow = s
	a, err := accounts.GetByID(context.Background(), 2)
	require.NoError(t, err)
	return s, store, accounts, transport, a
}
func astraBusinessRequest(t *testing.T, s *AstraBorrowService, a *Account) *http.Request {
	t.Helper()
	r, err := http.NewRequestWithContext(context.Background(), "POST", chatgptCodexURL, strings.NewReader(`{"model":"gpt-6-astra","input":"Business request","stream":true}`))
	require.NoError(t, err)
	r.Header, err = s.headers(r.Context(), a)
	require.NoError(t, err)
	r.Header.Set("x-codex-turn-state", "business-ticket")
	r.Header.Set("Cookie", "__oailb=stale-client-route; own=value")
	r.Header.Set(responsesLiteHeaderKey, "true")
	return r
}

func TestAstraBorrowEndToEndIdentityAndCache(t *testing.T) {
	s, store, _, transport, a := astraFixture(t)
	base := transport.fn
	calls := 0
	transport.fn = func(r *http.Request, proxy string, id int64) (*http.Response, error) {
		calls++
		if calls <= 3 {
			require.Empty(t, r.Header.Get(responsesLiteHeaderKey), "two-shot probes must use the full Responses protocol")
		}
		if id == 1 {
			require.Equal(t, "Bearer source-secret", r.Header.Get("Authorization"))
			require.Empty(t, r.Header.Get("Cookie"))
			require.Empty(t, r.Header.Get("x-codex-turn-state"))
		} else {
			require.Equal(t, "Bearer target-secret", r.Header.Get("Authorization"))
			require.Contains(t, r.Header.Get("Cookie"), "__oailb=route-secret")
			require.NotContains(t, r.Header.Get("Cookie"), "stale-client")
			require.NotEqual(t, "source-ticket", r.Header.Get("x-codex-turn-state"))
			if calls == 2 {
				require.Empty(t, r.Header.Get("x-codex-turn-state"))
			}
			if calls == 3 {
				require.Equal(t, "target-ticket", r.Header.Get("x-codex-turn-state"))
			}
			if calls >= 4 {
				require.Equal(t, "business-ticket", r.Header.Get("x-codex-turn-state"))
				require.Contains(t, r.Header.Get("Cookie"), "own=value")
			}
		}
		return base(r, proxy, id)
	}
	for i := 0; i < 2; i++ {
		req := astraBusinessRequest(t, s, a)
		resp, handled, err := s.RoundTrip(req, a, nil)
		require.NoError(t, err)
		require.True(t, handled)
		_ = resp.Body.Close()
		require.Contains(t, req.Header.Get("Cookie"), "stale-client", "do not mutate caller headers")
	}
	require.Equal(t, 5, calls, "one source + two target probes + two business requests")
	snap, err := s.Snapshot(context.Background())
	require.NoError(t, err)
	rows, err := store.ListAstraBorrowHistory(context.Background(), 0, 50)
	require.NoError(t, err)
	public, _ := json.Marshal([]any{snap, rows})
	for _, secret := range []string{"source-secret", "target-secret", "route-secret", "source-ticket", "target-ticket", "business-ticket"} {
		require.NotContains(t, string(public), secret)
	}
}

func TestAstraBorrowRejectsFailedProbeWithoutBusinessSend(t *testing.T) {
	for _, failure := range []string{"no_ticket", "new_ticket", "route_changed", "partial", "stream_failed", "wrong_model", "network"} {
		t.Run(failure, func(t *testing.T) {
			s, _, _, tr, a := astraFixture(t)
			base := tr.fn
			calls := 0
			tr.fn = func(req *http.Request, proxy string, id int64) (*http.Response, error) {
				calls++
				if id == 1 {
					return base(req, proxy, id)
				}
				if failure == "network" {
					return nil, errors.New("private proxy password")
				}
				cookie, ticket, body := "", "target-ticket", astraComplete
				switch failure {
				case "no_ticket":
					ticket = ""
				case "new_ticket":
					if calls == 3 {
						ticket = "replaced"
					}
				case "route_changed":
					cookie = "__oailb=other; Secure; Path=/; Max-Age=230"
				case "partial":
					body = `data: {"type":"response.created"}` + "\n\n"
				case "stream_failed":
					body = astraComplete + "data: {\"type\":\"response.failed\"}\n\n"
				case "wrong_model":
					body = strings.ReplaceAll(astraComplete, "gpt-6-astra", "gpt-6-luna")
				}
				return astraTestResponse(cookie, ticket, body), nil
			}
			resp, handled, err := s.RoundTrip(astraBusinessRequest(t, s, a), a, nil)
			require.Error(t, err)
			require.True(t, handled)
			require.Nil(t, resp)
			require.LessOrEqual(t, calls, 3)
			require.NotContains(t, err.Error(), "private")
			var typed *AstraBorrowRequestError
			require.ErrorAs(t, err, &typed)
		})
	}
}

func TestAstraBorrowForcedVerificationRevokesPass(t *testing.T) {
	s, _, _, tr, a := astraFixture(t)
	resp, _, err := s.RoundTrip(astraBusinessRequest(t, s, a), a, nil)
	require.NoError(t, err)
	_ = resp.Body.Close()
	tr.fn = func(*http.Request, string, int64) (*http.Response, error) { return nil, errors.New("failure") }
	require.Error(t, s.Verify(context.Background(), 2))
	s.mu.Lock()
	passed := s.checks[2].passed
	s.mu.Unlock()
	require.False(t, passed)
	snap, err := s.Snapshot(context.Background())
	require.NoError(t, err)
	require.Equal(t, "failed", snap.Statuses[1].State)
}

func TestAstraBorrowDisableExpiryAndIdentityChange(t *testing.T) {
	for _, change := range []string{"disable", "expire", "target_identity", "source_identity", "settings_error"} {
		t.Run(change, func(t *testing.T) {
			s, store, accounts, tr, a := astraFixture(t)
			resp, _, err := s.RoundTrip(astraBusinessRequest(t, s, a), a, nil)
			require.NoError(t, err)
			_ = resp.Body.Close()
			calls := 0
			tr.fn = func(*http.Request, string, int64) (*http.Response, error) { calls++; return nil, errors.New("stop") }
			switch change {
			case "disable":
				v, _ := s.Settings(context.Background())
				v.Enabled = false
				v.Revision = "off"
				store.config(v)
			case "expire":
				s.mu.Lock()
				r := s.routes[1]
				r.expires = time.Now().Add(-time.Second)
				s.routes[1] = r
				c := s.checks[2]
				c.route.expires = r.expires
				s.checks[2] = c
				s.mu.Unlock()
			case "target_identity":
				accounts.mu.Lock()
				accounts.items[2].Credentials["access_token"] = "new"
				accounts.mu.Unlock()
			case "source_identity":
				accounts.mu.Lock()
				accounts.items[1].Credentials["access_token"] = "new"
				accounts.mu.Unlock()
			case "settings_error":
				store.mu.Lock()
				store.err = errors.New("database down")
				store.mu.Unlock()
			}
			resp, handled, err := s.RoundTrip(astraBusinessRequest(t, s, a), a, nil)
			if change == "disable" {
				require.NoError(t, err)
				require.False(t, handled)
				require.Zero(t, calls)
			} else {
				require.Error(t, err)
				require.True(t, handled)
				require.Nil(t, resp)
				require.LessOrEqual(t, calls, 1)
			}
		})
	}
}

func TestAstraBorrowConfigChangeDuringProbe(t *testing.T) {
	s, store, _, tr, a := astraFixture(t)
	base := tr.fn
	calls := 0
	tr.fn = func(req *http.Request, proxy string, id int64) (*http.Response, error) {
		calls++
		if calls == 3 {
			v, _ := s.Settings(context.Background())
			v.Revision = "replaced"
			store.config(v)
		}
		return base(req, proxy, id)
	}
	_, handled, err := s.RoundTrip(astraBusinessRequest(t, s, a), a, nil)
	require.True(t, handled)
	require.ErrorContains(t, err, "configuration_changed")
	require.Equal(t, 3, calls)
}

func TestAstraBorrowBusinessTransportFailureNeverFailsOver(t *testing.T) {
	s, _, _, tr, a := astraFixture(t)
	base := tr.fn
	calls := 0
	tr.fn = func(req *http.Request, proxy string, id int64) (*http.Response, error) {
		calls++
		if calls == 4 {
			return nil, errors.New("connection refused, secret")
		}
		return base(req, proxy, id)
	}
	_, _, err := s.RoundTrip(astraBusinessRequest(t, s, a), a, nil)
	require.ErrorContains(t, err, "astra_business_transport_failed")
	require.Equal(t, 4, calls)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	err = s.gateway.handleOpenAIUpstreamTransportError(context.Background(), c, a, err, false)
	var failover *UpstreamFailoverError
	require.False(t, errors.As(err, &failover))
	require.Equal(t, 503, w.Code)
	require.NotContains(t, w.Body.String(), "secret")
}

func TestAstraBorrowCookieBoundaries(t *testing.T) {
	now := time.Now()
	for _, header := range []string{"__oailb=x; Path=/; Max-Age=200", "__oailb=x; Secure; Domain=evil.test; Max-Age=200", "__oailb=x; Secure; Path=/other; Max-Age=200", "__oailb=x; Secure; Path=/", "__oailb=x; Secure; Max-Age=0"} {
		resp := astraTestResponse(header, "", astraComplete)
		c, _ := astraBorrowCookie(resp.Cookies(), now, 230)
		require.Nil(t, c, header)
	}
	r := astraTestResponse("__oailb=x; Secure; Path=/; Max-Age=40", "", astraComplete)
	c, expires := astraBorrowCookie(r.Cookies(), now, 230)
	require.NotNil(t, c)
	require.Equal(t, now.Add(40*time.Second), expires)
}

func TestAstraBorrowSettingsValidationAndDisabledRead(t *testing.T) {
	s, store, _, tr, _ := astraFixture(t)
	tr.fn = func(*http.Request, string, int64) (*http.Response, error) {
		t.Fatal("read/config validation must not call upstream")
		return nil, nil
	}
	v := defaultAstraBorrowSettings()
	store.config(v)
	snap, err := s.Snapshot(context.Background())
	require.NoError(t, err)
	require.False(t, snap.Settings.Enabled)
	require.Empty(t, snap.Statuses)
	v.Enabled = true
	require.ErrorIs(t, v.Validate(), ErrAstraBorrowInvalid)
	v.SourceAccountIDs = []int64{1}
	v.TargetAccountIDs = []int64{1}
	require.ErrorIs(t, v.Validate(), ErrAstraBorrowInvalid)
	v.TargetAccountIDs = []int64{2}
	v.TTLSeconds = 241
	require.ErrorIs(t, v.Validate(), ErrAstraBorrowInvalid)
	v.TTLSeconds = 230
	require.NoError(t, v.Validate())
}

func TestAstraBorrowWebSocketGuardChecksLaterEnable(t *testing.T) {
	s, store, _, _, a := astraFixture(t)
	v, _ := s.Settings(context.Background())
	v.Enabled = false
	store.config(v)
	hooks := s.gateway.withAstraBorrowWSGuard(context.Background(), a, nil)
	require.NoError(t, hooks.BeforeRequest(1, nil, astraBorrowModel))
	v.Enabled = true
	v.Revision = "on"
	store.config(v)
	require.ErrorContains(t, hooks.BeforeRequest(2, nil, astraBorrowModel), "http_only")
}

func TestAstraBorrowFixedProxyAndTargetProxy(t *testing.T) {
	for _, follow := range []bool{true, false} {
		t.Run(map[bool]string{true: "follow_source", false: "target_proxy"}[follow], func(t *testing.T) {
			s, store, accounts, tr, _ := astraFixture(t)
			for id, host := range map[int64]string{1: "source.invalid", 2: "target.invalid"} {
				proxyID := id
				accounts.items[id].ProxyID = &proxyID
				accounts.items[id].Proxy = &Proxy{ID: id, Protocol: "http", Host: host, Port: 8080, Status: StatusActive, Username: "private", Password: "password"}
			}
			v, _ := s.Settings(context.Background())
			v.FollowSourceProxy = follow
			store.config(v)
			a, err := accounts.GetByID(context.Background(), 2)
			require.NoError(t, err)
			base := tr.fn
			tr.fn = func(req *http.Request, proxy string, id int64) (*http.Response, error) {
				if id == 1 || follow {
					require.Contains(t, proxy, "source.invalid")
				} else {
					require.Contains(t, proxy, "target.invalid")
				}
				return base(req, proxy, id)
			}
			resp, _, err := s.RoundTrip(astraBusinessRequest(t, s, a), a, nil)
			require.NoError(t, err)
			_ = resp.Body.Close()
			snap, err := s.Snapshot(context.Background())
			require.NoError(t, err)
			data, _ := json.Marshal(snap)
			require.NotContains(t, string(data), "password")
			require.NotContains(t, string(data), "invalid")
		})
	}
}

func TestAstraBorrowCachedTrafficDoesNotWaitForOtherProbes(t *testing.T) {
	s, _, _, _, a := astraFixture(t)
	resp, _, err := s.RoundTrip(astraBusinessRequest(t, s, a), a, nil)
	require.NoError(t, err)
	_ = resp.Body.Close()
	s.prepareMu.Lock()
	defer s.prepareMu.Unlock()
	resp, _, err = s.RoundTrip(astraBusinessRequest(t, s, a), a, nil)
	require.NoError(t, err)
	_ = resp.Body.Close()
}

func TestAstraBorrowSourceRevocationInvalidatesAllTargets(t *testing.T) {
	s, store, accounts, tr, a := astraFixture(t)
	v, _ := s.Settings(context.Background())
	v.TargetAccountIDs = []int64{2, 3}
	store.config(v)
	for _, id := range v.TargetAccountIDs {
		target, err := accounts.GetByID(context.Background(), id)
		require.NoError(t, err)
		resp, _, err := s.RoundTrip(astraBusinessRequest(t, s, target), target, nil)
		require.NoError(t, err)
		_ = resp.Body.Close()
	}
	tr.fn = func(*http.Request, string, int64) (*http.Response, error) {
		return astraTestResponse("__oailb=new; Secure; Max-Age=230", "", astraComplete), nil
	}
	resp, _, err := s.RoundTrip(astraBusinessRequest(t, s, a), a, nil)
	require.NoError(t, err)
	_ = resp.Body.Close()
	s.mu.Lock()
	require.Empty(t, s.checks)
	require.Empty(t, s.routes)
	s.mu.Unlock()
	tr.fn = func(*http.Request, string, int64) (*http.Response, error) {
		t.Fatal("revoked route must not send during cooldown")
		return nil, nil
	}
	target, _ := accounts.GetByID(context.Background(), 3)
	_, _, err = s.RoundTrip(astraBusinessRequest(t, s, target), target, nil)
	require.ErrorContains(t, err, "cooldown")
}

type astraSlotCache struct {
	ConcurrencyCache
	acquired []int64
}

func (s *astraSlotCache) AcquireAccountSlot(_ context.Context, id int64, _ int, _ string) (bool, error) {
	s.acquired = append(s.acquired, id)
	return id == 1, nil
}
func (*astraSlotCache) ReleaseAccountSlot(context.Context, int64, string) error { return nil }
func TestAstraBorrowUsesExistingBusinessConcurrencySlot(t *testing.T) {
	s, _, _, _, a := astraFixture(t)
	cache := &astraSlotCache{}
	s.gateway.concurrencyService = NewConcurrencyService(cache)
	resp, err := s.gateway.doOpenAIUpstream(astraBusinessRequest(t, s, a), "", a)
	require.NoError(t, err)
	_ = resp.Body.Close()
	require.Equal(t, []int64{1}, cache.acquired, "only source needs a new slot; target is already reserved by the handler")
}

func TestAstraBorrowNullListsAndSaveDisabled(t *testing.T) {
	s, store, _, tr, _ := astraFixture(t)
	store.raw = `{"enabled":false,"source_account_ids":null,"target_account_ids":null,"ttl_seconds":230}`
	v, err := s.Settings(context.Background())
	require.NoError(t, err)
	require.NotNil(t, v.SourceAccountIDs)
	require.NotNil(t, v.TargetAccountIDs)
	tr.fn = func(*http.Request, string, int64) (*http.Response, error) {
		t.Fatal("disabled save must not send probes")
		return nil, nil
	}
	saved, err := s.Save(context.Background(), v)
	require.NoError(t, err)
	require.NotEmpty(t, saved.Revision)
	require.NoError(t, s.RejectExcelTarget(context.Background(), 2))
}

func TestAstraBorrowRejectsUnsupportedHostAndKeepsOtherModels(t *testing.T) {
	s, _, _, tr, a := astraFixture(t)
	tr.fn = func(*http.Request, string, int64) (*http.Response, error) {
		t.Fatal("must not contact an unsupported host")
		return nil, nil
	}
	req := astraBusinessRequest(t, s, a)
	req.URL.Host = "evil.invalid"
	_, handled, err := s.RoundTrip(req, a, nil)
	require.True(t, handled)
	require.ErrorContains(t, err, "http_responses_only")
	req, _ = http.NewRequest("POST", chatgptCodexURL, strings.NewReader(`{"model":"other-model"}`))
	_, handled, err = s.RoundTrip(req, a, nil)
	require.NoError(t, err)
	require.False(t, handled)
}
