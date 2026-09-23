package service

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/Wei-Shaw/sub2api/internal/util/logredact"
)

// CandyMonitorService shares the existing fixed-puzzle probe, but owns independent
// scheduling and history. It never recovers/disables accounts based on answers.
type CandyMonitorService struct {
	repo      CandyMonitorRepository
	accounts  AccountRepository
	probe     func(context.Context, int64, string) (CandyTestResult, string, error)
	ctx       context.Context
	cancel    context.CancelFunc
	slots     chan struct{}
	mu        sync.Mutex
	stopped   bool
	wg        sync.WaitGroup
	startOnce sync.Once
}

func NewCandyMonitorService(repo CandyMonitorRepository, accounts AccountRepository, tester *AccountTestService) *CandyMonitorService {
	ctx, cancel := context.WithCancel(context.Background())
	return &CandyMonitorService{repo: repo, accounts: accounts, probe: tester.RunCandyTestBackground, ctx: ctx, cancel: cancel, slots: make(chan struct{}, 4)}
}
func (s *CandyMonitorService) Start() {
	s.startOnce.Do(func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		if s.stopped {
			return
		}
		s.wg.Add(1)
		go func() {
			defer s.wg.Done()
			ticker := time.NewTicker(30 * time.Second)
			defer ticker.Stop()
			for {
				s.tick()
				select {
				case <-s.ctx.Done():
					return
				case <-ticker.C:
				}
			}
		}()
	})
}
func (s *CandyMonitorService) Stop() {
	s.mu.Lock()
	s.stopped = true
	s.cancel()
	s.mu.Unlock()
	s.wg.Wait()
}
func validateCandyMonitorModel(model string) error {
	if model == "" || len(model) > 200 || strings.ContainsAny(model, "\r\n\x00") {
		return fmt.Errorf("%w: model is required (max 200 bytes)", ErrCandyMonitorInvalid)
	}
	// Share text-model restrictions without contacting upstream.
	if err := validateCandyTest(&Account{Platform: PlatformOpenAI}, model, AccountTestOptions{}); err != nil {
		return fmt.Errorf("%w: %v", ErrCandyMonitorInvalid, err)
	}
	return nil
}
func validateCandyInterval(n int) error {
	if n < 5 || n > 10080 {
		return fmt.Errorf("%w: interval must be 5–10080 minutes", ErrCandyMonitorInvalid)
	}
	return nil
}
func (s *CandyMonitorService) Settings(ctx context.Context) (*CandyMonitorSettings, error) {
	return s.repo.Settings(ctx)
}
func (s *CandyMonitorService) SaveSettings(ctx context.Context, c *CandyMonitorSettings) error {
	c.ModelID = strings.TrimSpace(c.ModelID)
	if err := validateCandyMonitorModel(c.ModelID); err != nil {
		return err
	}
	if err := validateCandyInterval(c.IntervalMinutes); err != nil {
		return err
	}
	if c.MaxResults < 10 || c.MaxResults > 500 {
		return fmt.Errorf("%w: history retention must be 10–500", ErrCandyMonitorInvalid)
	}
	return s.repo.SaveSettings(ctx, c)
}
func (s *CandyMonitorService) List(ctx context.Context, f CandyMonitorFilter) ([]CandyMonitorAccount, int64, error) {
	return s.repo.List(ctx, f)
}
func (s *CandyMonitorService) Configure(ctx context.Context, ids []int64, c CandyMonitorConfig) error {
	if len(ids) == 0 || len(ids) > 500 {
		return fmt.Errorf("%w: select 1–500 accounts", ErrCandyMonitorInvalid)
	}
	seen := map[int64]bool{}
	unique := make([]int64, 0, len(ids))
	for _, id := range ids {
		if id <= 0 {
			return ErrCandyMonitorInvalid
		}
		if !seen[id] {
			seen[id] = true
			unique = append(unique, id)
		}
	}
	c.ModelID = strings.TrimSpace(c.ModelID)
	if c.UseDefaults {
		defaults, err := s.repo.Settings(ctx)
		if err != nil {
			return err
		}
		c.ModelID = defaults.ModelID
		c.IntervalMinutes = defaults.IntervalMinutes
	}
	if err := validateCandyMonitorModel(c.ModelID); err != nil {
		return err
	}
	if err := validateCandyInterval(c.IntervalMinutes); err != nil {
		return err
	}
	accounts, err := s.accounts.GetByIDs(ctx, unique)
	if err != nil {
		return err
	}
	if len(accounts) != len(unique) {
		return fmt.Errorf("%w: one or more accounts no longer exist", ErrCandyMonitorInvalid)
	}
	for _, a := range accounts {
		if a.IsShadow() {
			return fmt.Errorf("%w: shadow accounts are not supported", ErrCandyMonitorInvalid)
		}
		if err := validateCandyTest(a, c.ModelID, AccountTestOptions{}); err != nil {
			return fmt.Errorf("%w: account %d: %v", ErrCandyMonitorInvalid, a.ID, err)
		}
	}
	return s.repo.Configure(ctx, unique, c)
}
func (s *CandyMonitorService) Queue(ctx context.Context, id int64, model string) (*CandyMonitorResult, error) {
	return s.queue(ctx, id, model, false)
}

// SetEnabled preserves each account's custom model and interval.
func (s *CandyMonitorService) SetEnabled(ctx context.Context, ids []int64, enabled bool) error {
	if len(ids) == 0 || len(ids) > 500 {
		return ErrCandyMonitorInvalid
	}
	for _, id := range ids {
		if id <= 0 {
			return ErrCandyMonitorInvalid
		}
	}
	return s.repo.SetEnabled(ctx, ids, enabled)
}
func (s *CandyMonitorService) queue(ctx context.Context, id int64, model string, scheduled bool) (*CandyMonitorResult, error) {
	model = strings.TrimSpace(model)
	if model == "" {
		model = CandyMonitorDefaultModel
	}
	if err := validateCandyMonitorModel(model); err != nil {
		return nil, err
	}
	a, err := s.accounts.GetByID(ctx, id)
	if err != nil {
		return nil, err
	}
	if a.IsShadow() {
		return nil, fmt.Errorf("%w: shadow accounts are not supported", ErrCandyMonitorInvalid)
	}
	if !scheduled {
		if err := validateCandyTest(a, model, AccountTestOptions{}); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrCandyMonitorInvalid, err)
		}
	}
	// Guard Add/Wait during shutdown; a database lease also excludes other instances.
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped {
		return nil, ErrCandyMonitorBusy
	}
	select {
	case s.slots <- struct{}{}:
	default:
		return nil, ErrCandyMonitorBusy
	}
	result, err := s.repo.Begin(ctx, id, model, scheduled)
	if err != nil {
		<-s.slots
		return nil, err
	}
	job := *result
	s.wg.Add(1)
	go func() { defer s.wg.Done(); defer func() { <-s.slots }(); s.execute(&job) }()
	return result, nil
}
func (s *CandyMonitorService) execute(result *CandyMonitorResult) {
	ctx, cancel := context.WithTimeout(s.ctx, 180*time.Second)
	defer cancel()
	verdict, text, err := s.probe(ctx, result.AccountID, result.ModelID)
	result.Verdict, result.Reason, result.Actual, result.Expected = verdict.Verdict, verdict.Reason, verdict.Actual, verdict.Expected
	result.Expected = 21
	if err != nil {
		result.Verdict, result.Reason, result.Actual = "inconclusive", "request_failed", nil
	}
	if result.Verdict == "" {
		result.Verdict = "inconclusive"
		result.Reason = "incomplete"
	}
	result.ResponseText = boundedCandyText(text, maxCandyOutputBytes)
	if err != nil {
		result.ErrorMessage = boundedCandyText(logredact.RedactText(err.Error()), 2048)
	}
	now := time.Now()
	result.FinishedAt = &now
	result.DurationMs = now.Sub(result.StartedAt).Milliseconds()
	// Save even after request/service cancellation, using a short independent deadline.
	saveCtx, saveCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer saveCancel()
	if err := s.repo.Finish(saveCtx, result); err != nil {
		slog.Error("candy monitor result save failed", "account_id", result.AccountID, "error", err)
	}
}
func boundedCandyText(text string, limit int) string {
	if len(text) <= limit {
		return text
	}
	text = text[:limit]
	for !utf8.ValidString(text) {
		text = text[:len(text)-1]
	}
	return text
}
func (s *CandyMonitorService) tick() {
	ctx, cancel := context.WithTimeout(s.ctx, 15*time.Second)
	defer cancel()
	plans, err := s.repo.Due(ctx, 4)
	if err != nil {
		if s.ctx.Err() == nil {
			slog.Warn("candy monitor scan failed", "error", err)
		}
		return
	}
	for _, p := range plans {
		if ctx.Err() != nil {
			return
		}
		if _, err := s.queue(ctx, p.AccountID, p.ModelID, true); err != nil && err != ErrCandyMonitorBusy {
			slog.Warn("candy monitor enqueue failed", "account_id", p.AccountID, "error", err)
		}
	}
}
func (s *CandyMonitorService) Result(ctx context.Context, id int64) (*CandyMonitorResult, error) {
	return s.repo.Result(ctx, id)
}
func (s *CandyMonitorService) History(ctx context.Context, id int64) ([]CandyMonitorResult, error) {
	return s.repo.History(ctx, id, 50)
}
