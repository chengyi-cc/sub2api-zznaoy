package service

import (
	"context"
	"errors"
	"time"
)

const CandyMonitorDefaultModel = "gpt-6-astra"

var ErrCandyMonitorBusy = errors.New("candy test already running or worker capacity reached")
var ErrCandyMonitorInvalid = errors.New("invalid candy monitor configuration")

type CandyMonitorSettings struct {
	Enabled         bool   `json:"enabled"`
	ModelID         string `json:"model_id"`
	IntervalMinutes int    `json:"interval_minutes"`
	MaxResults      int    `json:"max_results"`
}
type CandyMonitorConfig struct {
	Enabled         bool   `json:"enabled"`
	UseDefaults     bool   `json:"use_defaults"`
	ModelID         string `json:"model_id"`
	IntervalMinutes int    `json:"interval_minutes"`
}
type CandyMonitorResult struct {
	ID           int64      `json:"id"`
	AccountID    int64      `json:"account_id"`
	ModelID      string     `json:"model_id"`
	Source       string     `json:"source"`
	Verdict      string     `json:"verdict"`
	Reason       string     `json:"reason"`
	Actual       *int       `json:"actual,omitempty"`
	Expected     int        `json:"expected"`
	DurationMs   int64      `json:"duration_ms"`
	ResponseText string     `json:"response_text,omitempty"`
	ErrorMessage string     `json:"error_message,omitempty"`
	StartedAt    time.Time  `json:"started_at"`
	FinishedAt   *time.Time `json:"finished_at,omitempty"`
}
type CandyMonitorAccount struct {
	AccountID int64  `json:"account_id"`
	Name      string `json:"name"`
	Platform  string `json:"platform"`
	Status    string `json:"status"`
	Type      string `json:"type"`
	CandyMonitorConfig
	LastRunAt         *time.Time           `json:"last_run_at"`
	NextRunAt         *time.Time           `json:"next_run_at"`
	RunningUntil      *time.Time           `json:"running_until"`
	Latest            *CandyMonitorResult  `json:"latest"`
	History           []CandyMonitorResult `json:"history"`
	TotalTests        int64                `json:"total_tests"`
	Answer21Count     int64                `json:"answer_21_count"`
	Answer29Count     int64                `json:"answer_29_count"`
	OtherAnswerCount  int64                `json:"other_answer_count"`
	InconclusiveCount int64                `json:"inconclusive_count"`
}
type CandyMonitorFilter struct {
	GroupID     int64
	Search      string
	EnabledOnly bool
	Enabled     *bool
	Ungrouped   bool
	Platform    string
	Status      string
	Type        string
	PrivacyMode string
	Verdict     string
	Page        int
	PageSize    int
}
type CandyMonitorAccountState struct {
	AccountID int64 `json:"account_id"`
	CandyMonitorConfig
	LastValidAnswer *int       `json:"last_valid_answer"`
	LastValidAt     *time.Time `json:"last_valid_at"`
}
type CandyMonitorStates struct {
	SchedulerEnabled bool                       `json:"scheduler_enabled"`
	Items            []CandyMonitorAccountState `json:"items"`
}
type CandyMonitorRepository interface {
	AccountStates(context.Context, []int64) ([]CandyMonitorAccountState, error)
	SetMonitoring(context.Context, int64, bool) error
	Settings(context.Context) (*CandyMonitorSettings, error)
	SaveSettings(context.Context, *CandyMonitorSettings) error
	List(context.Context, CandyMonitorFilter) ([]CandyMonitorAccount, int64, error)
	Configure(context.Context, []int64, CandyMonitorConfig) error
	SetEnabled(context.Context, []int64, bool) error
	Due(context.Context, int) ([]CandyMonitorAccount, error)
	Begin(context.Context, int64, string, bool) (*CandyMonitorResult, error)
	Finish(context.Context, *CandyMonitorResult) error
	Result(context.Context, int64) (*CandyMonitorResult, error)
	History(context.Context, int64, int) ([]CandyMonitorResult, error)
}
