package service

import (
	"errors"
	"github.com/stretchr/testify/require"
	"io"
	"testing"
	"time"
)

func TestExcelBPSRawIdleClosesSilentRead(t *testing.T) {
	r, w := io.Pipe()
	defer w.Close()
	body := withExcelBPSIdleTimeout(r, 40*time.Millisecond)
	defer body.Close()
	started := time.Now()
	_, err := body.Read(make([]byte, 1))
	require.ErrorIs(t, err, errOpenAISSEIdle)
	require.Less(t, time.Since(started), time.Second)
}

func TestExcelBPSRawIdleAllowsFragmentsAndConsumerPauses(t *testing.T) {
	r, w := io.Pipe()
	body := withExcelBPSIdleTimeout(r, 100*time.Millisecond)
	defer body.Close()
	defer w.Close()
	go func() {
		for i := 0; i < 6; i++ {
			time.Sleep(20 * time.Millisecond)
			if _, err := w.Write([]byte("x")); err != nil {
				return
			}
		}
		w.Close()
	}()
	for i := 0; i < 6; i++ {
		buf := make([]byte, 1)
		n, err := body.Read(buf)
		require.NoError(t, err)
		require.Equal(t, 1, n)
		if i == 2 {
			time.Sleep(150 * time.Millisecond)
		}
	}
	_, err := body.Read(make([]byte, 1))
	require.True(t, errors.Is(err, io.EOF))
}

func TestExcelBPSIdleDisabledRetainsOriginalBody(t *testing.T) {
	r, w := io.Pipe()
	defer r.Close()
	defer w.Close()
	require.Same(t, r, withExcelBPSIdleTimeout(r, 0))
}
