//go:build unit

package service

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/stretchr/testify/require"
)

type candyAccountsStub struct {
	AccountRepository
	items map[int64]*Account
}

func (r *candyAccountsStub) GetByID(_ context.Context, id int64) (*Account, error) {
	a, ok := r.items[id]
	if !ok {
		return nil, errors.New("missing account")
	}
	return a, nil
}
func (r *candyAccountsStub) GetByIDs(_ context.Context, ids []int64) ([]*Account, error) {
	var out []*Account
	for _, id := range ids {
		if a := r.items[id]; a != nil {
			out = append(out, a)
		}
	}
	return out, nil
}

type candyMonitorStub struct {
	CandyMonitorRepository
	settings   CandyMonitorSettings
	configured CandyMonitorConfig
	ids        []int64
	mu         sync.Mutex
	active     map[int64]bool
	finished   chan CandyMonitorResult
	due        []CandyMonitorAccount
	states     []CandyMonitorAccountState
	monitoring map[int64]bool
}

func (r *candyMonitorStub) AccountStates(_ context.Context, ids []int64) ([]CandyMonitorAccountState, error) {
	items := []CandyMonitorAccountState{}
	for _, item := range r.states {
		for _, id := range ids {
			if item.AccountID == id {
				if enabled, ok := r.monitoring[id]; ok {
					item.Enabled = enabled
				}
				items = append(items, item)
			}
		}
	}
	return items, nil
}
func (r *candyMonitorStub) SetMonitoring(_ context.Context, id int64, enabled bool) error {
	if r.monitoring == nil {
		r.monitoring = map[int64]bool{}
	}
	r.monitoring[id] = enabled
	return nil
}

func TestCandyMonitorAccountRowToggle(t *testing.T) {
	s, r, accounts := newCandyMonitorTestService(t)
	answer := 29
	r.states = []CandyMonitorAccountState{{AccountID: 1, CandyMonitorConfig: CandyMonitorConfig{ModelID: "custom-text", IntervalMinutes: 17}, LastValidAnswer: &answer}}
	r.settings.Enabled = false
	state, err := s.SetMonitoring(context.Background(), 1, true)
	require.NoError(t, err)
	require.True(t, state.Items[0].Enabled)
	require.Equal(t, "custom-text", state.Items[0].ModelID)
	require.Equal(t, 17, state.Items[0].IntervalMinutes)
	require.Equal(t, 29, *state.Items[0].LastValidAnswer)
	require.False(t, state.SchedulerEnabled, "the row switch must not enable monitoring for all accounts")
	state, err = s.SetMonitoring(context.Background(), 1, false)
	require.NoError(t, err)
	require.False(t, state.Items[0].Enabled)
	require.Equal(t, 17, state.Items[0].IntervalMinutes)
	accounts.items[1].Platform = PlatformGrok
	_, err = s.SetMonitoring(context.Background(), 1, true)
	require.ErrorIs(t, err, ErrCandyMonitorInvalid)
	require.False(t, r.monitoring[1])
	for _, ids := range [][]int64{nil, {-1}, make([]int64, 501)} {
		_, err = s.AccountStates(context.Background(), ids)
		require.ErrorIs(t, err, ErrCandyMonitorInvalid)
	}
}

func (r *candyMonitorStub) Settings(context.Context) (*CandyMonitorSettings, error) {
	v := r.settings
	return &v, nil
}
func (r *candyMonitorStub) SaveSettings(_ context.Context, v *CandyMonitorSettings) error {
	r.settings = *v
	return nil
}
func (r *candyMonitorStub) Configure(_ context.Context, ids []int64, c CandyMonitorConfig) error {
	r.ids = ids
	r.configured = c
	return nil
}
func (r *candyMonitorStub) Due(context.Context, int) ([]CandyMonitorAccount, error) {
	return r.due, nil
}
func (r *candyMonitorStub) Begin(_ context.Context, id int64, model string, scheduled bool) (*CandyMonitorResult, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active[id] {
		return nil, ErrCandyMonitorBusy
	}
	r.active[id] = true
	source := "manual"
	if scheduled {
		source = "scheduled"
	}
	return &CandyMonitorResult{ID: id, AccountID: id, ModelID: model, Source: source, Verdict: "running", Expected: 21, StartedAt: time.Now()}, nil
}
func (r *candyMonitorStub) Finish(ctx context.Context, v *CandyMonitorResult) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	r.mu.Lock()
	r.active[v.AccountID] = false
	r.mu.Unlock()
	r.finished <- *v
	return nil
}
func newCandyMonitorTestService(t *testing.T) (*CandyMonitorService, *candyMonitorStub, *candyAccountsStub) {
	t.Helper()
	repo := &candyMonitorStub{settings: CandyMonitorSettings{Enabled: true, ModelID: CandyMonitorDefaultModel, IntervalMinutes: 60, MaxResults: 50}, active: map[int64]bool{}, finished: make(chan CandyMonitorResult, 10)}
	accounts := &candyAccountsStub{items: map[int64]*Account{}}
	for id := int64(1); id <= 6; id++ {
		accounts.items[id] = &Account{ID: id, Platform: PlatformOpenAI, Status: StatusActive, Schedulable: true}
	}
	s := NewCandyMonitorService(repo, accounts, &AccountTestService{})
	t.Cleanup(s.Stop)
	return s, repo, accounts
}
func TestCandyMonitorSkipsUnavailableScheduledAccounts(t *testing.T) {
	past, future := time.Now().Add(-time.Hour), time.Now().Add(time.Hour)
	for _, tc := range []struct {
		name   string
		change func(*Account)
	}{
		{"disabled", func(a *Account) { a.Status = "inactive" }},
		{"authentication error", func(a *Account) { a.Status = "error" }},
		{"scheduling disabled", func(a *Account) { a.Schedulable = false }},
		{"expired", func(a *Account) { a.AutoPauseOnExpired = true; a.ExpiresAt = &past }},
		{"cooldown", func(a *Account) { a.TempUnschedulableUntil = &future }},
		{"rate limited", func(a *Account) { a.RateLimitResetAt = &future }},
		{"overloaded", func(a *Account) { a.OverloadUntil = &future }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, r, accounts := newCandyMonitorTestService(t)
			tc.change(accounts.items[1])
			_, err := s.queue(context.Background(), 1, CandyMonitorDefaultModel, true)
			require.ErrorIs(t, err, ErrCandyMonitorBusy)
			require.Empty(t, r.active, "skip before claiming or creating a test")
			require.Empty(t, r.finished)
			// Manual probes remain available for troubleshooting the account.
			s.probe = func(context.Context, int64, string) (CandyTestResult, string, error) {
				return CandyTestResult{Verdict: "inconclusive"}, "", nil
			}
			_, err = s.Queue(context.Background(), 1, CandyMonitorDefaultModel)
			require.NoError(t, err)
			select {
			case v := <-r.finished:
				require.Equal(t, "manual", v.Source)
			case <-time.After(time.Second):
				t.Fatal("manual probe did not finish")
			}
		})
	}
}

func TestCandyMonitorConfigureValidatesWholeBatchAndInherits(t *testing.T) {
	s, r, accounts := newCandyMonitorTestService(t)
	require.NoError(t, s.Configure(context.Background(), []int64{1, 1, 2}, CandyMonitorConfig{Enabled: true, UseDefaults: true}))
	require.Equal(t, []int64{1, 2}, r.ids)
	require.Equal(t, CandyMonitorDefaultModel, r.configured.ModelID)
	require.Equal(t, 60, r.configured.IntervalMinutes)
	r.ids = nil
	accounts.items[2].Platform = PlatformGrok
	require.ErrorIs(t, s.Configure(context.Background(), []int64{1, 2}, CandyMonitorConfig{Enabled: true, UseDefaults: true}), ErrCandyMonitorInvalid)
	require.Nil(t, r.ids)
	require.ErrorIs(t, s.Configure(context.Background(), []int64{99}, CandyMonitorConfig{UseDefaults: true}), ErrCandyMonitorInvalid)
	override := CandyMonitorConfig{Enabled: true, ModelID: "custom-text", IntervalMinutes: 17}
	require.NoError(t, s.Configure(context.Background(), []int64{1}, override))
	require.Equal(t, override, r.configured)
}
func TestCandyMonitorSettingsRejectInvalidValues(t *testing.T) {
	s, _, _ := newCandyMonitorTestService(t)
	for _, v := range []CandyMonitorSettings{
		{ModelID: "gpt-image-1", IntervalMinutes: 60, MaxResults: 50},
		{ModelID: "text", IntervalMinutes: 4, MaxResults: 50},
		{ModelID: "text", IntervalMinutes: 10081, MaxResults: 50},
		{ModelID: "text", IntervalMinutes: 60, MaxResults: 9},
		{ModelID: "text", IntervalMinutes: 60, MaxResults: 501},
	} {
		require.ErrorIs(t, s.SaveSettings(context.Background(), &v), ErrCandyMonitorInvalid)
	}
}
func TestCandyMonitorRunsAfterRequestCancellationAndExcludesDuplicate(t *testing.T) {
	s, r, _ := newCandyMonitorTestService(t)
	release := make(chan struct{})
	t.Cleanup(func() {
		select {
		case <-release:
		default:
			close(release)
		}
	})
	started := make(chan context.Context, 1)
	s.probe = func(ctx context.Context, _ int64, model string) (CandyTestResult, string, error) {
		started <- ctx
		<-release
		answer := 21
		return CandyTestResult{Verdict: "pass", Actual: &answer, Expected: 21}, "CANDY_RESULT=21", nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	result, err := s.Queue(ctx, 1, "")
	require.NoError(t, err)
	require.Equal(t, CandyMonitorDefaultModel, result.ModelID)
	workerCtx := <-started
	cancel()
	require.NoError(t, workerCtx.Err())
	_, err = s.Queue(context.Background(), 1, "")
	require.ErrorIs(t, err, ErrCandyMonitorBusy)
	close(release)
	select {
	case saved := <-r.finished:
		require.Equal(t, "pass", saved.Verdict)
		require.Equal(t, "manual", saved.Source)
		require.Equal(t, 21, *saved.Actual)
	case <-time.After(time.Second):
		t.Fatal("result not persisted")
	}
	require.Equal(t, "running", result.Verdict, "returned response must not race with worker mutation")
}
func TestCandyMonitorCapacityShutdownAndInconclusive(t *testing.T) {
	s, r, _ := newCandyMonitorTestService(t)
	s.probe = func(ctx context.Context, _ int64, _ string) (CandyTestResult, string, error) {
		<-ctx.Done()
		return CandyTestResult{}, "partial", ctx.Err()
	}
	for id := int64(1); id <= 4; id++ {
		_, err := s.Queue(context.Background(), id, "")
		require.NoError(t, err)
	}
	_, err := s.Queue(context.Background(), 5, "")
	require.ErrorIs(t, err, ErrCandyMonitorBusy)
	s.Stop()
	for i := 0; i < 4; i++ {
		select {
		case v := <-r.finished:
			require.Equal(t, "inconclusive", v.Verdict)
			require.Equal(t, 21, v.Expected)
			require.Nil(t, v.Actual)
			require.NotEmpty(t, v.ErrorMessage)
		default:
			t.Fatal("shutdown lost result")
		}
	}
	_, err = s.Queue(context.Background(), 6, "")
	require.ErrorIs(t, err, ErrCandyMonitorBusy)
}
func TestCandyMonitorScheduledInvalidMappingDoesNotStarveQueue(t *testing.T) {
	s, r, accounts := newCandyMonitorTestService(t)
	accounts.items[1].Credentials = map[string]any{"model_mapping": map[string]any{CandyMonitorDefaultModel: "gpt-image-1"}}
	r.due = []CandyMonitorAccount{{AccountID: 1, CandyMonitorConfig: CandyMonitorConfig{ModelID: CandyMonitorDefaultModel}}}
	s.probe = func(context.Context, int64, string) (CandyTestResult, string, error) {
		return CandyTestResult{}, "", errors.New("model mapping changed")
	}
	s.tick()
	select {
	case v := <-r.finished:
		require.Equal(t, "scheduled", v.Source)
		require.Equal(t, "inconclusive", v.Verdict)
	case <-time.After(time.Second):
		t.Fatal("invalid mapping starved due queue")
	}
}
func TestCandyMonitorBoundedUTF8(t *testing.T) {
	result := boundedCandyText(strings.Repeat("糖", 100), 64)
	require.LessOrEqual(t, len(result), 64)
	require.True(t, utf8.ValidString(result))
}
