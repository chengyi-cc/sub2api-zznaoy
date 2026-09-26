package errorarchive

import (
	"context"
	"encoding/json"
	"sync"
)

type traceKey struct{}
type Diagnostic struct {
	Phase     string `json:"phase"`
	Request   []byte `json:"prepared_request_base64,omitempty"`
	Response  []byte `json:"upstream_response_base64,omitempty"`
	Truncated bool   `json:"truncated"`
}
type Trace struct {
	mu        sync.Mutex
	remaining int
	items     []Diagnostic
	slots     chan struct{}
	reserved  bool
	closed    bool
}

func (c *Capture) WithTrace(ctx context.Context) (context.Context, *Trace) {
	t := &Trace{remaining: 2 << 20, slots: c.store.traceSlots}
	c.trace = t
	return context.WithValue(ctx, traceKey{}, t), t
}

func WithTrace(ctx context.Context) (context.Context, *Trace) {
	t := &Trace{remaining: 2 << 20}
	return context.WithValue(ctx, traceKey{}, t), t
}
func HasTrace(ctx context.Context) bool { _, ok := ctx.Value(traceKey{}).(*Trace); return ok }
func AddDiagnostic(ctx context.Context, phase string, request, response []byte) {
	t, _ := ctx.Value(traceKey{}).(*Trace)
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed || t.remaining == 0 || len(t.items) >= 6 {
		return
	}
	// Reserve heavy diagnostics only when a failure is observed, never for the
	// entire lifetime of each healthy stream.
	if t.slots != nil && !t.reserved {
		select {
		case t.slots <- struct{}{}:
			t.reserved = true
		default:
			t.remaining = 0
			t.items = append(t.items, Diagnostic{Phase: phase, Truncated: true})
			return
		}
	}
	d := Diagnostic{Phase: phase}
	copyPart := func(raw []byte, max int) []byte {
		if max > t.remaining {
			max = t.remaining
		}
		if len(raw) > max {
			raw = raw[:max]
			d.Truncated = true
		}
		t.remaining -= len(raw)
		return append([]byte(nil), raw...)
	}
	// Preserve the invalid tool response first; long input must not crowd it out.
	d.Response = copyPart(response, 512<<10)
	d.Request = copyPart(request, 768<<10)
	t.items = append(t.items, d)
}
func (t *Trace) Snapshot() json.RawMessage {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if len(t.items) == 0 {
		return nil
	}
	raw, _ := json.Marshal(t.items)
	return raw
}

func (t *Trace) Release() {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.closed = true
	t.items = nil
	if t.reserved {
		<-t.slots
		t.reserved = false
	}
}
