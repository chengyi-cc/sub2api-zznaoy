package service

import (
	"io"
	"sync"
	"sync/atomic"
	"time"
)

// Time only a blocked raw upstream Read. Tool argument fragments count as
// activity even while the bridge withholds them. Downstream backpressure and
// keepalive writes cannot create or reset an upstream read deadline.
type excelBPSIdleBody struct {
	body      io.ReadCloser
	idle      time.Duration
	closeOnce sync.Once
	closeErr  error
}

func withExcelBPSIdleTimeout(body io.ReadCloser, idle time.Duration) io.ReadCloser {
	if idle <= 0 {
		return body
	}
	return &excelBPSIdleBody{body: body, idle: idle}
}

func (b *excelBPSIdleBody) Close() error {
	b.closeOnce.Do(func() { b.closeErr = b.body.Close() })
	return b.closeErr
}

func (b *excelBPSIdleBody) Read(p []byte) (int, error) {
	var state atomic.Int32 // 0: reading, 1: read finished, 2: timed out.
	timer := time.AfterFunc(b.idle, func() {
		if state.CompareAndSwap(0, 2) {
			_ = b.Close()
		}
	})
	n, err := b.body.Read(p)
	finished := state.CompareAndSwap(0, 1)
	timer.Stop()
	if !finished {
		return n, errOpenAISSEIdle
	}
	return n, err
}
