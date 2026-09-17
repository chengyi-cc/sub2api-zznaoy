package turnstate

import (
	"context"
	"net/http"
	"sync"
	"time"
)

type Runtime struct {
	updateMu sync.Mutex
	mu       sync.RWMutex
	manager  *Manager
	build    func(Config) (*Manager, error)
	closed   bool
}

func NewRuntime(build func(Config) (*Manager, error)) *Runtime {
	return &Runtime{build: build}
}

func (runtime *Runtime) Update(config Config, persist func() error) error {
	runtime.updateMu.Lock()
	defer runtime.updateMu.Unlock()
	if runtime.closed {
		return context.Canceled
	}
	candidate, err := runtime.build(config)
	if err != nil {
		return err
	}
	if persist != nil {
		if err = persist(); err != nil {
			candidate.Close()
			return err
		}
	}
	previous := runtime.manager
	if previous != nil {
		previous.Close()
	}
	if previous != nil && candidate != nil {
		previous.mu.Lock()
		candidate.mu.Lock()
		for key, entry := range previous.targets {
			if time.Since(entry.lastSeen) <= activeWindow {
				copy := *entry
				copy.headers = entry.headers.Clone()
				copy.running = false
				copy.retryAt = time.Time{}
				copy.status.LastError = ""
				copy.status.RetryAt = nil
				candidate.targets[key] = &copy
			}
		}
		candidate.schedulePendingLocked()
		candidate.mu.Unlock()
		previous.mu.Unlock()
	}
	runtime.mu.Lock()
	runtime.manager = candidate
	runtime.mu.Unlock()
	return nil
}

func (runtime *Runtime) Apply(ctx context.Context, accountID int64, model string, headers http.Header, options ...Options) bool {
	if runtime == nil {
		return false
	}
	runtime.mu.RLock()
	defer runtime.mu.RUnlock()
	if runtime.manager == nil {
		return false
	}
	return runtime.manager.Apply(ctx, accountID, model, headers, options...)
}

func (runtime *Runtime) Force(ctx context.Context, accountID int64, model string, headers http.Header, options Options) bool {
	if runtime == nil {
		return false
	}
	runtime.mu.RLock()
	defer runtime.mu.RUnlock()
	if runtime.manager == nil {
		return false
	}
	return runtime.manager.Force(ctx, accountID, model, headers, options)
}

func (runtime *Runtime) Forget(accountID int64) {
	if runtime == nil {
		return
	}
	runtime.mu.RLock()
	defer runtime.mu.RUnlock()
	runtime.manager.Forget(accountID)
}

func (runtime *Runtime) Configured(source string) bool {
	if runtime == nil {
		return false
	}
	runtime.mu.RLock()
	defer runtime.mu.RUnlock()
	return runtime.manager != nil && runtime.manager.Configured(source)
}

func (runtime *Runtime) Countries() []string {
	if runtime == nil {
		return []string{}
	}
	runtime.mu.RLock()
	defer runtime.mu.RUnlock()
	if runtime.manager == nil {
		return []string{}
	}
	return runtime.manager.Countries()
}

func (runtime *Runtime) Inspect(ctx context.Context, accountID int64, options Options) []Status {
	if runtime == nil {
		return []Status{}
	}
	runtime.mu.RLock()
	defer runtime.mu.RUnlock()
	if runtime.manager == nil {
		return []Status{}
	}
	return runtime.manager.Inspect(ctx, accountID, options)
}

func (runtime *Runtime) History(ctx context.Context, accountID int64) ([]Attempt, error) {
	if runtime == nil {
		return []Attempt{}, nil
	}
	runtime.mu.RLock()
	defer runtime.mu.RUnlock()
	if runtime.manager == nil {
		return []Attempt{}, nil
	}
	return runtime.manager.History(ctx, accountID)
}

func (runtime *Runtime) Close() {
	if runtime == nil {
		return
	}
	runtime.updateMu.Lock()
	defer runtime.updateMu.Unlock()
	runtime.mu.Lock()
	defer runtime.mu.Unlock()
	runtime.closed = true
	if runtime.manager != nil {
		runtime.manager.Close()
	}
}
