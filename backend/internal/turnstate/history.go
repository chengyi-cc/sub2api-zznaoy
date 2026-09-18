package turnstate

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/redis/go-redis/v9"
)

type Attempt struct {
	Options
	At            time.Time `json:"at"`
	Model         string    `json:"model"`
	Country       string    `json:"country,omitempty"`
	ActualCountry string    `json:"actual_country,omitempty"`
	SourceIP      string    `json:"source_ip,omitempty"`
	Status        int       `json:"status"`
	Length        int       `json:"length"`
	Accepted      bool      `json:"accepted"`
	Error         string    `json:"error,omitempty"`
	DurationMS    int64     `json:"duration_ms"`
}

type countryRotation struct {
	Index    int `json:"index"`
	Failures int `json:"failures"`
}

func rotationKey(key string, options Options) string {
	return key + ":rotation:" + options.Profile + ":" + options.Source
}

func (manager *Manager) readRotation(ctx context.Context, key string, options Options) (countryRotation, error) {
	if options.Source != SourcePurchased {
		return countryRotation{}, nil
	}
	data, err := manager.cache.Get(ctx, rotationKey(key, options)).Bytes()
	if errors.Is(err, redis.Nil) {
		return countryRotation{}, nil
	}
	if err != nil {
		return countryRotation{}, err
	}
	var rotation countryRotation
	if json.Unmarshal(data, &rotation) != nil || rotation.Index < 0 || rotation.Failures < 0 || rotation.Failures >= 3 {
		return countryRotation{}, errors.New("invalid country rotation")
	}
	return rotation, nil
}

func (manager *Manager) saveRotation(ctx context.Context, key string, options Options, rotation countryRotation) error {
	if options.Source != SourcePurchased {
		return nil
	}
	data, _ := json.Marshal(rotation)
	return manager.cache.Set(ctx, rotationKey(key, options), data, 7*24*time.Hour).Err()
}

func historyKey(accountID int64) string { return fmt.Sprintf("codex:turn-state:history:%d", accountID) }
func accountIndexKey(accountID int64) string {
	return fmt.Sprintf("codex:turn-state:models:%d", accountID)
}

func (manager *Manager) recordAttempt(parent context.Context, accountID int64, model string, options Options, country string, started time.Time, record Record, failure error) {
	entry := Attempt{Options: options, At: started.UTC(), Model: model, Country: country, ActualCountry: record.Country, SourceIP: record.SourceIP, Status: 200, Length: len(record.Value), Accepted: failure == nil, DurationMS: time.Since(started).Milliseconds()}
	if failure != nil {
		entry.Error = failure.Error()
		entry.Status = 0
		var rejected *ProbeError
		if errors.As(failure, &rejected) {
			entry.Status, entry.Length, entry.SourceIP, entry.ActualCountry = rejected.Status, rejected.Length, rejected.SourceIP, rejected.Country
		}
	}
	data, _ := json.Marshal(entry)
	ctx, cancel := context.WithTimeout(context.WithoutCancel(parent), 2*time.Second)
	defer cancel()
	key := historyKey(accountID)
	pipeline := manager.cache.TxPipeline()
	pipeline.LPush(ctx, key, data)
	pipeline.LTrim(ctx, key, 0, 199)
	pipeline.Expire(ctx, key, 7*24*time.Hour)
	_, _ = pipeline.Exec(ctx)
}

func (manager *Manager) History(ctx context.Context, accountID int64) ([]Attempt, error) {
	result := []Attempt{}
	if manager == nil {
		return result, nil
	}
	values, err := manager.cache.LRange(ctx, historyKey(accountID), 0, 199).Result()
	if err != nil {
		return result, errors.New("detection history cache unavailable")
	}
	for _, value := range values {
		var entry Attempt
		if json.Unmarshal([]byte(value), &entry) == nil && !entry.At.Before(time.Now().Add(-7*24*time.Hour)) {
			result = append(result, entry)
		}
	}
	return result, nil
}

func (manager *Manager) updateStatusRecord(status *Status, record Record) {
	issued, expires, refresh := record.IssuedAt, record.ExpiresAt, record.ExpiresAt.Add(-manager.refreshBefore())
	status.Options = record.Options
	status.IssuedAt, status.ExpiresAt, status.RefreshAt = &issued, &expires, &refresh
	status.Length, status.Country, status.SourceIP = len(record.Value), record.Country, record.SourceIP
}

func (manager *Manager) Inspect(ctx context.Context, accountID int64, options Options) []Status {
	result := []Status{}
	if manager == nil {
		return result
	}
	options = options.Normalized()
	byModel := map[string]Status{}
	for _, status := range manager.Snapshot(accountID) {
		if status.Options == options && !manager.ModelExcluded(status.Model) {
			byModel[status.Model] = status
		}
	}
	models, _ := manager.cache.SMembers(ctx, accountIndexKey(accountID)).Result()
	for _, model := range models {
		if manager.ModelExcluded(model) {
			continue
		}
		if len(byModel) >= 256 {
			break
		}
		record, err := manager.read(ctx, accountID, model)
		if err != nil || record.Options != options {
			continue
		}
		status, exists := byModel[model]
		if !exists {
			status = Status{Model: model, State: "ready"}
		}
		manager.updateStatusRecord(&status, record)
		byModel[model] = status
	}
	for _, status := range byModel {
		result = append(result, status)
	}
	sort.Slice(result, func(first, second int) bool { return result[first].Model < result[second].Model })
	return result
}

func (manager *Manager) Countries() []string {
	if manager == nil {
		return []string{}
	}
	return append([]string{}, manager.config.Countries...)
}
