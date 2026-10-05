package service

// Fixed-egress adaptation of ranxi2001/sub2api v2.9.8 (PR #288).
// Only __oailb is borrowed. OAuth credentials and turn-state tickets are never shared.
import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"sync"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/google/uuid"
	"github.com/tidwall/gjson"
)

const astraBorrowSettingKey = "astra_gateway_borrow_v1"
const astraBorrowModel = "gpt-6-astra"

var ErrAstraBorrowInvalid = errors.New("invalid Astra borrowing configuration")
var ErrAstraBorrowBusy = errors.New("astra_borrow_busy")

// Distinguish borrowing failures from generic transport failures. They must not
// trigger automatic account failover or replay a possibly delivered request.
type AstraBorrowRequestError struct{ Code string }

func (e *AstraBorrowRequestError) Error() string { return e.Code }

type AstraBorrowSettings struct {
	Enabled           bool    `json:"enabled"`
	SourceAccountIDs  []int64 `json:"source_account_ids"`
	TargetAccountIDs  []int64 `json:"target_account_ids"`
	FollowSourceProxy bool    `json:"follow_source_proxy"`
	TTLSeconds        int     `json:"ttl_seconds"`
	Revision          string  `json:"revision"`
}

func defaultAstraBorrowSettings() AstraBorrowSettings {
	return AstraBorrowSettings{SourceAccountIDs: []int64{}, TargetAccountIDs: []int64{}, FollowSourceProxy: true, TTLSeconds: 230}
}

func normalizeAstraBorrowSettings(v AstraBorrowSettings) AstraBorrowSettings {
	if v.SourceAccountIDs == nil {
		v.SourceAccountIDs = []int64{}
	}
	if v.TargetAccountIDs == nil {
		v.TargetAccountIDs = []int64{}
	}
	return v
}

func (v AstraBorrowSettings) Validate() error {
	if v.TTLSeconds < 30 || v.TTLSeconds > 240 {
		return fmt.Errorf("%w: ttl must be 30–240 seconds", ErrAstraBorrowInvalid)
	}
	seen := map[int64]bool{}
	for _, ids := range [][]int64{v.SourceAccountIDs, v.TargetAccountIDs} {
		if len(ids) > 20 {
			return fmt.Errorf("%w: at most 20 accounts per role", ErrAstraBorrowInvalid)
		}
		for _, id := range ids {
			if id <= 0 || seen[id] {
				return fmt.Errorf("%w: duplicate, overlapping or invalid account", ErrAstraBorrowInvalid)
			}
			seen[id] = true
		}
	}
	if v.Enabled && (len(v.SourceAccountIDs) == 0 || len(v.TargetAccountIDs) == 0) {
		return fmt.Errorf("%w: select sources and targets", ErrAstraBorrowInvalid)
	}
	return nil
}

type AstraBorrowHistory struct {
	ID              int64     `json:"id"`
	SourceAccountID int64     `json:"source_account_id"`
	TargetAccountID int64     `json:"target_account_id"`
	Passed          bool      `json:"passed"`
	Reason          string    `json:"reason"`
	CheckedAt       time.Time `json:"checked_at"`
}

type AstraBorrowStore interface {
	GetValue(context.Context, string) (string, error)
	Set(context.Context, string, string) error
	AppendAstraBorrowHistory(context.Context, AstraBorrowHistory) error
	ListAstraBorrowHistory(context.Context, int64, int) ([]AstraBorrowHistory, error)
}

type AstraBorrowStatus struct {
	Probe           *AstraBorrowProbeDetails `json:"probe,omitempty"`
	AccountID       int64                    `json:"account_id"`
	SourceAccountID int64                    `json:"source_account_id"`
	State           string                   `json:"state"`
	Reason          string                   `json:"reason"`
	CheckedAt       time.Time                `json:"checked_at"`
	ExpiresAt       *time.Time               `json:"expires_at,omitempty"`
}

type AstraBorrowSnapshot struct {
	Settings     AstraBorrowSettings `json:"settings"`
	Statuses     []AstraBorrowStatus `json:"statuses"`
	Preparing    bool                `json:"preparing"`
	HistoryError bool                `json:"history_error"`
}

type astraBorrowRoute struct {
	cookie         http.Cookie
	expires        time.Time
	proxy          string
	sourceID       int64
	sourceIdentity [32]byte
}
type astraBorrowValidation struct {
	route          astraBorrowRoute
	targetIdentity [32]byte
	retryAfter     time.Time
	passed         bool
}

// No autonomous ticker: saving prepares once; an expired route is renewed on demand.
// State is process-local; config is re-read before publishing or using evidence.
type AstraBorrowService struct {
	gateway      *OpenAIGatewayService
	store        AstraBorrowStore
	mu           sync.Mutex
	prepareMu    sync.Mutex
	saveMu       sync.Mutex
	revision     string
	routes       map[int64]astraBorrowRoute
	checks       map[int64]astraBorrowValidation
	statuses     map[int64]AstraBorrowStatus
	sourceRetry  map[int64]time.Time
	preparing    bool
	cancel       context.CancelFunc
	historyError bool
}

func NewAstraBorrowService(gateway *OpenAIGatewayService, store AstraBorrowStore) *AstraBorrowService {
	return &AstraBorrowService{gateway: gateway, store: store, routes: map[int64]astraBorrowRoute{}, checks: map[int64]astraBorrowValidation{}, statuses: map[int64]AstraBorrowStatus{}, sourceRetry: map[int64]time.Time{}}
}

func (s *AstraBorrowService) Settings(ctx context.Context) (AstraBorrowSettings, error) {
	v := defaultAstraBorrowSettings()
	raw, err := s.store.GetValue(ctx, astraBorrowSettingKey)
	if errors.Is(err, ErrSettingNotFound) {
		return v, nil
	}
	if err != nil {
		return v, errors.New("astra_settings_unavailable")
	}
	if raw != "" {
		if json.Unmarshal([]byte(raw), &v) != nil || v.Validate() != nil {
			return v, errors.New("astra_settings_invalid")
		}
	}
	return normalizeAstraBorrowSettings(v), nil
}

func (s *AstraBorrowService) syncRevision(v AstraBorrowSettings) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.revision != v.Revision {
		s.revision = v.Revision
		s.routes = map[int64]astraBorrowRoute{}
		s.checks = map[int64]astraBorrowValidation{}
		s.statuses = map[int64]AstraBorrowStatus{}
		s.preparing = false
		// Keep failure cooldowns across edits to avoid retry storms.
		for id := range s.sourceRetry {
			if !slices.Contains(v.SourceAccountIDs, id) {
				delete(s.sourceRetry, id)
			}
		}
	}
}

func (s *AstraBorrowService) account(ctx context.Context, id int64) (*Account, error) {
	a, err := s.gateway.accountRepo.GetByID(ctx, id)
	if err != nil || a == nil || !a.IsOpenAIOAuth() || a.IsShadow() || a.IsOpenAIAgentIdentity() || a.IsOpenAIPersonalAccessToken() || !a.IsSchedulable() {
		return nil, errors.New("astra_account_unavailable")
	}
	if a.IsExcelBPSEnabled() {
		return nil, errors.New("astra_disable_excel_first")
	}
	if a.ProxyID != nil && (a.Proxy == nil || !a.Proxy.IsActive() || a.Proxy.IsExpired(time.Now())) {
		return nil, errors.New("astra_proxy_unavailable")
	}
	if a.GetMappedModel(astraBorrowModel) != astraBorrowModel || !a.IsModelSupported(astraBorrowModel) {
		return nil, errors.New("astra_model_unavailable")
	}
	return a, nil
}

func astraAccountProxy(a *Account) string {
	if a.ProxyID != nil && a.Proxy != nil {
		return a.Proxy.URL()
	}
	return ""
}

func astraAccountIdentity(a *Account, headers http.Header, profile *tlsfingerprint.Profile) [32]byte {
	// Compare identity/config in memory only; never publish this hash or its input.
	data, _ := json.Marshal([]any{a.Credentials, a.Extra, a.ProxyID, astraAccountProxy(a), a.Status, a.Schedulable, a.ExpiresAt, a.GroupIDs, headers.Get("Authorization"), headers.Get("ChatGPT-Account-ID"), headers.Get("User-Agent"), headers.Get("Originator"), headers.Get("Version"), profile})
	return sha256.Sum256(data)
}

func (s *AstraBorrowService) headers(ctx context.Context, a *Account) (http.Header, error) {
	token, _, err := s.gateway.GetAccessToken(ctx, a)
	if err != nil || token == "" {
		return nil, errors.New("astra_auth_failed")
	}
	h := http.Header{"Authorization": {"Bearer " + token}}
	if err := resolveAndSetOpenAIChatGPTAccountHeaders(ctx, s.gateway.accountRepo, h, a); err != nil {
		return nil, errors.New("astra_auth_failed")
	}
	applyAstraBorrowProbeIdentity(h)
	return h, nil
}

func (s *AstraBorrowService) Save(ctx context.Context, v AstraBorrowSettings) (AstraBorrowSettings, error) {
	s.saveMu.Lock()
	defer s.saveMu.Unlock()
	v = normalizeAstraBorrowSettings(v)
	if err := v.Validate(); err != nil {
		return v, err
	}
	if v.Enabled {
		for _, id := range append(slices.Clone(v.SourceAccountIDs), v.TargetAccountIDs...) {
			if _, err := s.account(ctx, id); err != nil {
				return v, fmt.Errorf("%w: account #%d: %s", ErrAstraBorrowInvalid, id, err)
			}
		}
	}
	v.Revision = uuid.NewString()
	data, _ := json.Marshal(v)
	if err := s.store.Set(ctx, astraBorrowSettingKey, string(data)); err != nil {
		return v, errors.New("astra_save_failed")
	}
	s.mu.Lock()
	if s.cancel != nil {
		s.cancel()
	}
	s.mu.Unlock()
	s.syncRevision(v)
	if v.Enabled {
		s.startPreparation(v)
	}
	return v, nil
}

func (s *AstraBorrowService) startPreparation(v AstraBorrowSettings) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	s.mu.Lock()
	if s.cancel != nil {
		s.cancel()
	}
	s.cancel = cancel
	s.preparing = true
	s.mu.Unlock()
	go func() {
		defer cancel()
		defer func() {
			s.mu.Lock()
			if s.revision == v.Revision {
				s.preparing = false
			}
			s.mu.Unlock()
		}()
		for _, id := range v.TargetAccountIDs {
			if ctx.Err() != nil {
				return
			}
			for {
				err := s.verify(ctx, v, id, true)
				if !errors.Is(err, ErrAstraBorrowBusy) {
					break
				}
				select {
				case <-ctx.Done():
					return
				case <-time.After(100 * time.Millisecond):
				}
			}
		}
	}()
}

func (s *AstraBorrowService) Verify(ctx context.Context, id int64) error {
	v, err := s.Settings(ctx)
	if err != nil {
		return err
	}
	if !v.Enabled || !slices.Contains(v.TargetAccountIDs, id) {
		return fmt.Errorf("%w: target not enabled", ErrAstraBorrowInvalid)
	}
	s.syncRevision(v)
	return s.verify(ctx, v, id, true)
}

func (s *AstraBorrowService) verify(ctx context.Context, v AstraBorrowSettings, id int64, force bool) error {
	if force {
		s.mu.Lock()
		delete(s.checks, id)
		s.mu.Unlock()
	}
	a, err := s.account(ctx, id)
	if err != nil {
		s.record(v, 0, id, false, err.Error(), nil)
		return err
	}
	h, err := s.headers(ctx, a)
	if err != nil {
		s.record(v, 0, id, false, err.Error(), nil)
		return err
	}
	profile := resolveOpenAITransportTLSProfile(s.gateway.tlsFPProfileService, a)
	_, err = s.ensureRoute(ctx, v, a, h, profile, force)
	return err
}

func (s *AstraBorrowService) current(ctx context.Context, v AstraBorrowSettings) bool {
	if ctx.Err() != nil {
		return false
	}
	now, err := s.Settings(ctx)
	return err == nil && now.Enabled && now.Revision == v.Revision
}

func (s *AstraBorrowService) record(v AstraBorrowSettings, source, target int64, passed bool, reason string, expiry *time.Time, evidence ...*AstraBorrowProbeDetails) {
	now := time.Now()
	s.mu.Lock()
	if s.revision != v.Revision {
		s.mu.Unlock()
		return
	}
	id := target
	if id == 0 {
		id = source
	}
	state := "failed"
	if passed {
		state = "ready"
	}
	var probe *AstraBorrowProbeDetails
	if len(evidence) != 0 && evidence[0] != nil {
		copy := *evidence[0]
		probe = &copy
	}
	s.statuses[id] = AstraBorrowStatus{Probe: probe, AccountID: id, SourceAccountID: source, State: state, Reason: reason, CheckedAt: now, ExpiresAt: expiry}
	s.mu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	err := s.store.AppendAstraBorrowHistory(ctx, AstraBorrowHistory{SourceAccountID: source, TargetAccountID: target, Passed: passed, Reason: reason, CheckedAt: now})
	s.mu.Lock()
	s.historyError = err != nil
	s.mu.Unlock()
}

func (s *AstraBorrowService) Snapshot(ctx context.Context) (AstraBorrowSnapshot, error) {
	v, err := s.Settings(ctx)
	if err != nil {
		return AstraBorrowSnapshot{}, err
	}
	s.syncRevision(v)
	out := AstraBorrowSnapshot{Settings: v, Statuses: []AstraBorrowStatus{}}
	s.mu.Lock()
	defer s.mu.Unlock()
	out.Preparing, out.HistoryError = s.preparing, s.historyError
	for _, id := range append(slices.Clone(v.SourceAccountIDs), v.TargetAccountIDs...) {
		row, ok := s.statuses[id]
		if !ok {
			row = AstraBorrowStatus{AccountID: id, State: "not_tested", Reason: "not_tested"}
		}
		if !v.Enabled {
			row.State = "disabled"
		} else if row.ExpiresAt != nil && !time.Now().Before(*row.ExpiresAt) {
			row.State = "expired"
		}
		out.Statuses = append(out.Statuses, row)
	}
	return out, nil
}

func (s *AstraBorrowService) History(ctx context.Context, before int64) ([]AstraBorrowHistory, error) {
	return s.store.ListAstraBorrowHistory(ctx, before, 50)
}

func (s *AstraBorrowService) ensureRoute(ctx context.Context, v AstraBorrowSettings, target *Account, headers http.Header, profile *tlsfingerprint.Profile, force bool) (astraBorrowRoute, error) {
	var zero astraBorrowRoute
	identity := astraAccountIdentity(target, headers, profile)
	// Verified traffic does not queue behind probes for unrelated targets.
	if !force && s.current(ctx, v) {
		s.mu.Lock()
		prior, exists := s.checks[target.ID]
		revision := s.revision
		s.mu.Unlock()
		if revision == v.Revision && exists && prior.passed && prior.targetIdentity == identity && time.Now().Before(prior.route.expires) && s.routeStored(prior.route) && s.routeIdentityValid(ctx, prior.route) {
			return prior.route, nil
		}
	}
	if !s.prepareMu.TryLock() {
		return zero, ErrAstraBorrowBusy
	}
	defer s.prepareMu.Unlock()
	ctx, cancel := context.WithTimeout(ctx, 150*time.Second)
	defer cancel()
	if !s.current(ctx, v) {
		return zero, errors.New("astra_configuration_changed")
	}
	s.syncRevision(v)
	s.mu.Lock()
	prior, exists := s.checks[target.ID]
	s.mu.Unlock()
	if exists && !force && prior.targetIdentity == identity && time.Now().Before(prior.route.expires) {
		if prior.passed && s.routeStored(prior.route) && s.routeIdentityValid(ctx, prior.route) {
			return prior.route, nil
		}
		if !prior.passed && time.Now().Before(prior.retryAfter) {
			return zero, errors.New("astra_probe_cooldown")
		}
	}
	// A forced probe immediately revokes the old pass, including while it is running.
	s.mu.Lock()
	delete(s.checks, target.ID)
	s.statuses[target.ID] = AstraBorrowStatus{AccountID: target.ID, State: "verifying", Reason: "verifying", CheckedAt: time.Now()}
	s.mu.Unlock()
	reason := "astra_no_source_route"
	lastRecordedTargetReason := ""
	for _, sourceID := range v.SourceAccountIDs {
		if ctx.Err() != nil {
			reason = "astra_probe_timeout"
			break
		}
		route, err := s.sourceRoute(ctx, v, sourceID)
		if err != nil {
			reason = err.Error()
			continue
		}
		proxy := astraAccountProxy(target)
		if v.FollowSourceProxy {
			proxy = route.proxy
		}
		details, probeErr := s.probeTarget(ctx, target, headers, profile, route, proxy)
		err = probeErr
		if err == nil && (!s.current(ctx, v) || !time.Now().Before(route.expires) || !s.routeStored(route) || !s.routeIdentityValid(ctx, route)) {
			err = errors.New("astra_configuration_changed")
		}
		if err == nil {
			latest, latestErr := s.account(ctx, target.ID)
			if latestErr != nil || astraAccountIdentity(latest, headers, profile) != identity {
				err = errors.New("astra_account_changed")
			}
		}
		passed := err == nil
		reason = "astra_probe_passed"
		if err != nil {
			reason = err.Error()
		}
		s.mu.Lock()
		if s.revision == v.Revision {
			s.checks[target.ID] = astraBorrowValidation{route: route, targetIdentity: identity, passed: passed, retryAfter: time.Now().Add(30 * time.Second)}
		}
		s.mu.Unlock()
		s.record(v, sourceID, target.ID, passed, reason, &route.expires, &details)
		lastRecordedTargetReason = reason
		if passed {
			return route, nil
		}
	}
	// Do not duplicate a recorded target failure with an anonymous summary row.
	if lastRecordedTargetReason != reason {
		s.record(v, 0, target.ID, false, reason, nil)
	}
	return zero, errors.New(reason)
}

func (s *AstraBorrowService) routeIdentityValid(ctx context.Context, route astraBorrowRoute) bool {
	a, err := s.account(ctx, route.sourceID)
	return err == nil && astraAccountIdentity(a, http.Header{}, nil) == route.sourceIdentity
}

func (s *AstraBorrowService) routeStored(route astraBorrowRoute) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	current, ok := s.routes[route.sourceID]
	return ok && current.cookie.Value == route.cookie.Value && current.expires.Equal(route.expires) && current.sourceIdentity == route.sourceIdentity
}

func (s *AstraBorrowService) sourceRoute(ctx context.Context, v AstraBorrowSettings, id int64) (astraBorrowRoute, error) {
	s.mu.Lock()
	route, exists := s.routes[id]
	retry := s.sourceRetry[id]
	s.mu.Unlock()
	if exists && time.Now().Before(route.expires) && s.routeIdentityValid(ctx, route) {
		return route, nil
	}
	if time.Now().Before(retry) {
		return astraBorrowRoute{}, errors.New("astra_source_cooldown")
	}
	a, err := s.account(ctx, id)
	if err == nil {
		var headers http.Header
		headers, err = s.headers(ctx, a)
		if err == nil {
			var shot astraBorrowShot
			shot, err = s.fire(ctx, a, headers, resolveOpenAITransportTLSProfile(s.gateway.tlsFPProfileService, a), astraAccountProxy(a), "", "")
			if err == nil {
				var cookie *http.Cookie
				cookie, route.expires = astraBorrowCookie(shot.cookies, shot.receivedAt, v.TTLSeconds)
				if cookie == nil || !time.Now().Before(route.expires) {
					err = errors.New("astra_source_cookie_missing")
				} else {
					route.cookie, route.proxy, route.sourceID = *cookie, astraAccountProxy(a), id
					route.sourceIdentity = astraAccountIdentity(a, http.Header{}, nil)
				}
			}
		}
	}
	if err == nil && !s.current(ctx, v) {
		err = errors.New("astra_configuration_changed")
	}
	if err != nil {
		s.mu.Lock()
		if s.revision == v.Revision {
			delete(s.routes, id)
			s.sourceRetry[id] = time.Now().Add(30 * time.Second)
		}
		s.mu.Unlock()
		s.record(v, id, 0, false, err.Error(), nil)
		return astraBorrowRoute{}, err
	}
	s.mu.Lock()
	if s.revision == v.Revision {
		s.routes[id] = route
	}
	s.mu.Unlock()
	s.record(v, id, 0, true, "astra_source_ready", &route.expires)
	return route, nil
}

// RoundTrip reports handled=true on every configured target Astra request, including
// failures. Callers must never fall back to an unverified direct route.
func (s *AstraBorrowService) RoundTrip(req *http.Request, account *Account, profile *tlsfingerprint.Profile) (result *http.Response, handled bool, resultErr error) {
	sent := false
	defer func() {
		if handled && resultErr != nil {
			code := resultErr.Error()
			if sent {
				code = "astra_business_transport_failed"
			}
			resultErr = &AstraBorrowRequestError{Code: code}
		}
	}()
	v, err := s.Settings(req.Context())
	if err != nil {
		return nil, true, err
	}
	if !v.Enabled || account == nil || !slices.Contains(v.TargetAccountIDs, account.ID) {
		return nil, false, nil
	}
	if req.GetBody == nil {
		return nil, true, errors.New("astra_request_not_inspectable")
	}
	r, err := req.GetBody()
	if err != nil {
		return nil, true, errors.New("astra_request_not_inspectable")
	}
	// The gateway has already enforced its configured body limit. Do not add a
	// smaller hidden limit that would also break unrelated models on this account.
	body, readErr := io.ReadAll(r)
	_ = r.Close()
	if readErr != nil || !gjson.ValidBytes(body) {
		return nil, true, errors.New("astra_request_not_inspectable")
	}
	model := gjson.GetBytes(body, "model").String()
	if model == "" {
		return nil, true, errors.New("astra_request_not_inspectable")
	}
	if model != astraBorrowModel {
		return nil, false, nil
	}
	if req.URL == nil || req.URL.Scheme != "https" || req.URL.Host != "chatgpt.com" || (req.Host != "" && req.Host != "chatgpt.com") || req.Method != http.MethodPost || (req.URL.Path != "/backend-api/codex/responses" && req.URL.Path != "/backend-api/codex/responses/lite") {
		return nil, true, errors.New("astra_http_responses_only")
	}
	latest, err := s.account(req.Context(), account.ID)
	if err != nil {
		return nil, true, err
	}
	if astraAccountIdentity(latest, req.Header, profile) != astraAccountIdentity(account, req.Header, profile) {
		return nil, true, errors.New("astra_account_changed")
	}
	route, err := s.ensureRoute(req.Context(), v, latest, req.Header, profile, false)
	if err != nil {
		return nil, true, err
	}
	if !s.current(req.Context(), v) || !time.Now().Before(route.expires) || !s.routeStored(route) {
		return nil, true, errors.New("astra_configuration_changed")
	}
	proxy := astraAccountProxy(latest)
	if v.FollowSourceProxy {
		proxy = route.proxy
	}
	req = req.Clone(WithHTTPUpstreamRedirectsDisabled(req.Context()))
	astraSetRouteCookie(req.Header, route.cookie.Value)
	req.GetBody = nil // A possibly delivered business request must never be rewound.
	sent = true
	resp, err := s.gateway.httpUpstream.DoWithTLS(req, proxy, latest.ID, latest.Concurrency, profile)
	if err == nil && (resp == nil || resp.Body == nil) {
		err = errors.New("astra_empty_business_response")
	}
	// Business requests are sent once. A changed route revokes future use; do not
	// replay or alter the response body, which the existing billing path owns.
	if err != nil || resp == nil || resp.StatusCode >= 400 || astraRouteChanged(resp, route.cookie.Value) {
		s.mu.Lock()
		invalidated := false
		// An old in-flight failure must not revoke a newly validated route.
		if current, ok := s.routes[route.sourceID]; s.revision == v.Revision && ok && current.cookie.Value == route.cookie.Value && current.expires.Equal(route.expires) {
			invalidated = true
			delete(s.routes, route.sourceID)
			s.sourceRetry[route.sourceID] = time.Now().Add(30 * time.Second)
			for id, check := range s.checks {
				if check.route.sourceID == route.sourceID && check.route.cookie.Value == route.cookie.Value {
					delete(s.checks, id)
					row := s.statuses[id]
					row.State, row.Reason, row.ExpiresAt = "failed", "astra_business_route_invalidated", nil
					s.statuses[id] = row
				}
			}
			row := s.statuses[route.sourceID]
			row.State, row.Reason, row.ExpiresAt = "failed", "astra_business_route_invalidated", nil
			s.statuses[route.sourceID] = row
		}
		s.mu.Unlock()
		if invalidated {
			s.record(v, route.sourceID, latest.ID, false, "astra_business_route_invalidated", nil)
		}
	}
	return resp, true, err
}

func (s *AstraBorrowService) RejectWebSocket(ctx context.Context, accountID int64) error {
	v, err := s.Settings(ctx)
	if err != nil {
		return err
	}
	if v.Enabled && slices.Contains(v.TargetAccountIDs, accountID) {
		return errors.New("astra_borrow_http_only: use HTTP /v1/responses for borrowing targets")
	}
	return nil
}

func (s *AstraBorrowService) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cancel != nil {
		s.cancel()
	}
	s.routes = map[int64]astraBorrowRoute{}
	s.checks = map[int64]astraBorrowValidation{}
}
