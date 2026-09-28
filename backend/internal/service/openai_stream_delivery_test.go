package service

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type openAIExplicitFlushErrorWriter struct {
	gin.ResponseWriter
	called bool
}

func (w *openAIExplicitFlushErrorWriter) FlushError() error {
	w.called = true
	return io.ErrClosedPipe
}

type openAIOpaqueFlushWriter struct {
	gin.ResponseWriter
	called bool
}

func (w *openAIOpaqueFlushWriter) Flush() {
	w.called = true
	w.ResponseWriter.Flush()
}

func TestFlushOpenAIStreamPreservesHeadersAndWrapperSemantics(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Run("Gin unwraps the error-capable transport", func(t *testing.T) {
		recorder := newOpenAIResponseFlushRecorder()
		recorder.flushError = io.ErrClosedPipe
		c, _ := gin.CreateTestContext(recorder)
		c.Status(http.StatusAccepted)
		require.ErrorIs(t, flushOpenAIStream(c.Writer), io.ErrClosedPipe)
		require.True(t, c.Writer.Written())
		require.Equal(t, http.StatusAccepted, recorder.status)
		require.Equal(t, 1, recorder.flushErrorCalls)
	})
	t.Run("explicit wrapper flush takes precedence", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(recorder)
		w := &openAIExplicitFlushErrorWriter{ResponseWriter: c.Writer}
		require.ErrorIs(t, flushOpenAIStream(w), io.ErrClosedPipe)
		require.True(t, w.called)
		require.False(t, recorder.Flushed)
	})
	t.Run("opaque wrapper retains its own flush", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(recorder)
		w := &openAIOpaqueFlushWriter{ResponseWriter: c.Writer}
		require.NoError(t, flushOpenAIStream(w))
		require.True(t, w.called)
		require.True(t, recorder.Flushed)
	})
	t.Run("normal flush succeeds", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(recorder)
		require.NoError(t, flushOpenAIStream(c.Writer))
		require.True(t, recorder.Flushed)
	})
}
