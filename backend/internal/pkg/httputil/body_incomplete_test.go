package httputil

import (
	"io"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

type incompleteBodyObserver struct {
	io.ReadCloser
	err error
}

func (b *incompleteBodyObserver) OnBodyReadError(err error) { b.err = err }

func TestReadRequestBodyRejectsShortUploadEvenWhenPrefixIsValidJSON(t *testing.T) {
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"test"}`))
	req.ContentLength += 100
	observer := &incompleteBodyObserver{ReadCloser: req.Body}
	req.Body = observer
	body, err := ReadRequestBodyWithPrealloc(req)
	require.ErrorIs(t, err, io.ErrUnexpectedEOF)
	require.Nil(t, body, "partial requests must never be forwarded")
	require.ErrorIs(t, observer.err, io.ErrUnexpectedEOF)
}
