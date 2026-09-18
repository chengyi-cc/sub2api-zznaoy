package turnstate

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"
)

var errInvalidated = errors.New("cached state invalidated by upstream response")

func rejectedValueKey(accountID int64, model, identity, value string, options Options) string {
	digest := sha256.Sum256([]byte(identity + "\x00" + options.Profile + "\x00" + options.Source + "\x00" + value))
	return fmt.Sprintf("%s:rejected:%x", recordKey(accountID, model), digest[:])
}

func (manager *Manager) ObserveResponse(parent context.Context, accountID int64, model string, sent http.Header, received http.Header, status int, options Options) bool {
	if manager == nil || !*manager.config.RefreshOnRejection || accountID <= 0 {
		return false
	}
	model = strings.TrimSpace(model)
	options = options.Normalized()
	length := len(strings.TrimSpace(received.Get(Header)))
	if model == "" || len(model) > 256 || manager.ModelExcluded(model) || !((options.Profile == ProfilePro && length == 312) || (options.Profile == ProfileTeam && length == 356)) {
		return false
	}
	value := strings.TrimSpace(sent.Get(Header))
	if len(value) != acceptedLength(options.Profile) {
		return false
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(parent), 250*time.Millisecond)
	defer cancel()
	key := recordKey(accountID, model)
	data, err := manager.cache.Get(ctx, key).Bytes()
	if err != nil {
		return false
	}
	var record Record
	if json.Unmarshal(data, &record) != nil {
		return false
	}
	if record.Profile == "" {
		record.Profile = ProfilePro
	}
	if record.Source == "" {
		record.Source = SourceIPv6
	}
	if record.Invalidated || record.Value != value || record.Model != model || record.Options != options || record.Identity != accountIdentity(sent) || !record.ExpiresAt.After(time.Now()) {
		return false
	}
	record.Invalidated = true
	invalidated, err := json.Marshal(record)
	if err != nil {
		return false
	}
	rejectedKey := rejectedValueKey(accountID, model, record.Identity, value, options)
	changed, err := manager.cache.Eval(ctx, `
if redis.call('get', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('set', KEYS[2], '1', 'PX', ARGV[3])
redis.call('set', KEYS[1], ARGV[2], 'PX', ARGV[3])
return 1`, []string{key, rejectedKey}, data, invalidated, max(int64(1), time.Until(record.ExpiresAt).Milliseconds())).Int()
	if err != nil || changed != 1 {
		return false
	}
	forceCtx, forceCancel := context.WithTimeout(manager.ctx, 250*time.Millisecond)
	manager.Force(forceCtx, accountID, model, sent.Clone(), options)
	forceCancel()
	manager.appendHistory(ctx, accountID, Attempt{Options: options, At: time.Now().UTC(), Model: model, Kind: "response_rejection", Status: status, Length: length,
		Error: fmt.Sprintf("业务响应状态头长度%d触发重采集，已停用本次请求对应的旧值", length)})
	return true
}

func (manager *Manager) saveAcquiredRecord(ctx context.Context, accountID int64, record Record) error {
	encoded, err := json.Marshal(record)
	if err != nil {
		return errors.New("cannot encode acquired state")
	}
	rejectedKey := rejectedValueKey(accountID, record.Model, record.Identity, record.Value, record.Options)
	saved, err := manager.cache.Eval(ctx, `
if redis.call('exists', KEYS[2]) == 1 then return 0 end
redis.call('set', KEYS[1], ARGV[1], 'PX', ARGV[2])
return 1`, []string{recordKey(accountID, record.Model), rejectedKey}, encoded, max(int64(1), time.Until(record.ExpiresAt).Milliseconds())).Int()
	if err != nil {
		return errors.New("cannot save acquired state")
	}
	if saved != 1 {
		return &ProbeError{Message: "candidate was invalidated by an upstream response", Length: len(record.Value), Status: 200, Country: record.Country, SourceIP: record.SourceIP}
	}
	return nil
}

func (manager *Manager) stripRejectedValue(ctx context.Context, accountID int64, model string, headers http.Header, options Options) {
	value := strings.TrimSpace(headers.Get(Header))
	if len(value) != acceptedLength(options.Profile) {
		return
	}
	key := rejectedValueKey(accountID, model, accountIdentity(headers), value, options)
	if rejected, err := manager.cache.Exists(ctx, key).Result(); err == nil && rejected > 0 {
		for name := range headers {
			if strings.EqualFold(name, Header) {
				delete(headers, name)
			}
		}
	}
}

func clearInvalidatedStatus(status *Status) {
	status.State = "unavailable"
	status.IssuedAt, status.ExpiresAt, status.RefreshAt = nil, nil, nil
	status.Length, status.Country, status.SourceIP = 0, "", ""
	status.LastError = errInvalidated.Error()
}
