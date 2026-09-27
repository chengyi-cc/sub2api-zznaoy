package service

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/usagestats"
	"github.com/stretchr/testify/require"
)

type lifetimeStatsRepo struct {
	usageBatchLogRepoStub
	mu                    sync.Mutex
	today, lifetime       map[int64]*usagestats.AccountStats
	todayErr, lifetimeErr error
	singleCalls           int
}

func (r *lifetimeStatsRepo) read(ctx context.Context, id int64, lifetime bool) (*usagestats.AccountStats, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	stats, err := r.today, r.todayErr
	if lifetime {
		stats, err = r.lifetime, r.lifetimeErr
	}
	if err != nil {
		return nil, err
	}
	if value := stats[id]; value != nil {
		return value, nil
	}
	return &usagestats.AccountStats{}, nil
}

func (r *lifetimeStatsRepo) GetAccountTodayStats(ctx context.Context, id int64) (*usagestats.AccountStats, error) {
	r.mu.Lock()
	r.singleCalls++
	r.mu.Unlock()
	return r.read(ctx, id, false)
}

func (r *lifetimeStatsRepo) GetAccountWindowStats(ctx context.Context, id int64, start time.Time) (*usagestats.AccountStats, error) {
	r.mu.Lock()
	r.singleCalls++
	r.mu.Unlock()
	return r.read(ctx, id, start.IsZero())
}

type lifetimeStatsBatchRepo struct {
	lifetimeStatsRepo
	batchCalls     int
	failTodayBatch bool
}

func (r *lifetimeStatsBatchRepo) GetAccountWindowStatsBatch(ctx context.Context, ids []int64, start time.Time) (map[int64]*usagestats.AccountStats, error) {
	r.batchCalls++
	if r.failTodayBatch && !start.IsZero() {
		return nil, errors.New("batch unavailable")
	}
	result := make(map[int64]*usagestats.AccountStats, len(ids))
	for _, id := range ids {
		value, err := r.read(ctx, id, start.IsZero())
		if err != nil {
			return nil, err
		}
		result[id] = value
	}
	return result, nil
}

func lifetimeStatsFixture() lifetimeStatsRepo {
	return lifetimeStatsRepo{
		today:    map[int64]*usagestats.AccountStats{3: {Requests: 10, Tokens: 1000, Cost: 1.25, StandardCost: 2.5, UserCost: 0.75}},
		lifetime: map[int64]*usagestats.AccountStats{3: {Requests: 50, Tokens: 800000000, Cost: 1904.56}},
	}
}

func TestAccountTodayStatsLifetimePreservesTodayAndFailureSemantics(t *testing.T) {
	for _, failLifetime := range []bool{false, true} {
		r := lifetimeStatsFixture()
		if failLifetime {
			r.lifetimeErr = errors.New("lifetime query unavailable")
		}
		svc := &AccountUsageService{usageLogRepo: &r}
		got, err := svc.GetTodayStats(context.Background(), 3)
		require.NoError(t, err)
		require.EqualValues(t, 1000, got.Tokens)
		require.Equal(t, 1.25, got.Cost)
		require.Equal(t, 2.5, got.StandardCost)
		require.Equal(t, 0.75, got.UserCost)
		require.EqualValues(t, 10, got.Requests)
		if failLifetime {
			require.Nil(t, got.LifetimeTokens)
			require.Nil(t, got.LifetimeCost)
			raw, err := json.Marshal(got)
			require.NoError(t, err)
			require.NotContains(t, string(raw), "lifetime_")
		} else {
			require.NotNil(t, got.LifetimeTokens)
			require.NotNil(t, got.LifetimeCost)
			require.EqualValues(t, 800000000, *got.LifetimeTokens)
			require.Equal(t, 1904.56, *got.LifetimeCost)
		}
	}
	r := lifetimeStatsFixture()
	r.todayErr = errors.New("today query failed")
	svc := &AccountUsageService{usageLogRepo: &r}
	got, err := svc.GetTodayStats(context.Background(), 3)
	require.Error(t, err)
	require.Nil(t, got)
	require.Equal(t, 1, r.singleCalls)
}

func TestAccountTodayStatsLifetimeSuccessfulZeroIsNotMissing(t *testing.T) {
	r := lifetimeStatsFixture()
	svc := &AccountUsageService{usageLogRepo: &r}
	got, err := svc.GetTodayStats(context.Background(), 8)
	require.NoError(t, err)
	require.NotNil(t, got.LifetimeTokens)
	require.NotNil(t, got.LifetimeCost)
	raw, err := json.Marshal(got)
	require.NoError(t, err)
	var fields map[string]any
	require.NoError(t, json.Unmarshal(raw, &fields))
	require.Equal(t, float64(0), fields["lifetime_tokens"])
	require.Equal(t, float64(0), fields["lifetime_cost"])
}

func TestAccountTodayStatsBatchLifetimeUsesTwoQueriesAndBoundedFallback(t *testing.T) {
	for _, mode := range []string{"batch", "batch lifetime failure", "batch fallback", "no batch"} {
		t.Run(mode, func(t *testing.T) {
			batch := &lifetimeStatsBatchRepo{lifetimeStatsRepo: lifetimeStatsFixture()}
			var repo UsageLogRepository = batch
			switch mode {
			case "batch lifetime failure":
				batch.lifetimeErr = errors.New("unavailable")
			case "batch fallback":
				batch.failTodayBatch = true
			case "no batch":
				repo = &batch.lifetimeStatsRepo
			}
			svc := &AccountUsageService{usageLogRepo: repo}
			got, err := svc.GetTodayStatsBatch(context.Background(), []int64{3, 8, 3, 0, -1})
			require.NoError(t, err)
			require.Len(t, got, 2)
			require.EqualValues(t, 1000, got[3].Tokens)
			require.Equal(t, 1.25, got[3].Cost)
			if mode == "batch lifetime failure" {
				require.Nil(t, got[3].LifetimeTokens)
				require.Nil(t, got[8].LifetimeCost)
			} else {
				require.NotNil(t, got[3].LifetimeTokens)
				require.EqualValues(t, 800000000, *got[3].LifetimeTokens)
				require.NotNil(t, got[8].LifetimeCost)
				require.Zero(t, *got[8].LifetimeCost)
			}
			if mode == "batch" || mode == "batch lifetime failure" {
				require.Equal(t, 2, batch.batchCalls)
				require.Zero(t, batch.singleCalls, "successful batch reads must not query per account")
			} else {
				require.Equal(t, 4, batch.singleCalls, "fallback queries each account only once per range")
			}
			beforeBatch, beforeSingle := batch.batchCalls, batch.singleCalls
			empty, err := svc.GetTodayStatsBatch(context.Background(), []int64{0, -1})
			require.NoError(t, err)
			require.Empty(t, empty)
			require.Equal(t, beforeBatch, batch.batchCalls)
			require.Equal(t, beforeSingle, batch.singleCalls)
		})
	}
}
