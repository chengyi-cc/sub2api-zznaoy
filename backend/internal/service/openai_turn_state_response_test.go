package service

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/Wei-Shaw/sub2api/internal/turnstate"
	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
)

type turnStateResponseUpstream struct {
	response    *http.Response
	accountID   int64
	concurrency int
	calls       int
}

func (upstream *turnStateResponseUpstream) Do(_ *http.Request, _ string, accountID int64, concurrency int) (*http.Response, error) {
	upstream.calls++
	upstream.accountID, upstream.concurrency = accountID, concurrency
	return upstream.response, nil
}

func (upstream *turnStateResponseUpstream) DoWithTLS(request *http.Request, proxy string, accountID int64, concurrency int, _ *tlsfingerprint.Profile) (*http.Response, error) {
	return upstream.Do(request, proxy, accountID, concurrency)
}

type turnStateResponseBody struct {
	reader io.Reader
	reads  int
	closed bool
}

func (body *turnStateResponseBody) Read(buffer []byte) (int, error) {
	body.reads++
	return body.reader.Read(buffer)
}

func (body *turnStateResponseBody) Close() error {
	body.closed = true
	return nil
}

func TestTurnStateAutoResponseKeepsStreamingBodyAndTransportConcurrency(test *testing.T) {
	for _, transport := range []string{"business", "account-test"} {
		test.Run(transport, func(test *testing.T) {
			ctx := context.Background()
			cache := redis.NewClient(&redis.Options{Addr: miniredis.RunT(test).Addr()})
			defer cache.Close()
			runtime := turnstate.NewRuntime(func(config turnstate.Config) (*turnstate.Manager, error) {
				return turnstate.New(config, cache, func(context.Context, int64, http.Header) (http.Header, turnstate.Options, bool) {
					return nil, turnstate.Options{}, false
				})
			})
			defer runtime.Close()
			require.NoError(test, runtime.Update(turnstate.Config{ProxyHost: "proxy.example:7778", ProxyUsername: "test_{country}_{session}", ProxyPassword: "test"}, nil))
			raw := make([]byte, 217)
			raw[0] = 0x80
			binary.BigEndian.PutUint64(raw[1:9], uint64(time.Now().Add(-5*time.Minute).Unix()))
			record, err := turnstate.Parse(base64.URLEncoding.EncodeToString(raw), "gpt-6-astra", time.Now(), turnstate.ProfilePro)
			require.NoError(test, err)
			payload, err := json.Marshal(record)
			require.NoError(test, err)
			key := fmt.Sprintf("codex:turn-state:v1:42:%x", sha256.Sum256([]byte(record.Model)))
			require.NoError(test, cache.Set(ctx, key, payload, time.Hour).Err())
			body := &turnStateResponseBody{reader: strings.NewReader("data: hello\n\ndata: [DONE]\n\n")}
			response := &http.Response{StatusCode: 200, Header: make(http.Header), Body: body}
			response.Header.Set(turnstate.Header, strings.Repeat("x", 312))
			upstream := &turnStateResponseUpstream{response: response}
			gateway := &OpenAIGatewayService{turnStateAuto: runtime, httpUpstream: upstream}
			account := &Account{ID: 42, Platform: PlatformOpenAI, Type: AccountTypeSetupToken, Concurrency: 17, Extra: map[string]any{turnstate.EnabledKey: true, turnstate.ProfileKey: turnstate.ProfilePro}}
			request, err := http.NewRequest(http.MethodPost, "https://chatgpt.com/backend-api/codex/responses", strings.NewReader(`{"model":"gpt-6-astra","stream":true}`))
			require.NoError(test, err)
			var returned *http.Response
			if transport == "business" {
				returned, err = gateway.doOpenAIUpstream(request, "", account)
			} else {
				service := &AccountTestService{openaiGatewayService: gateway, httpUpstream: upstream}
				returned, err = service.doOpenAIAccountTestUpstream(request, "", account, true)
			}
			require.NoError(test, err)
			require.Same(test, response, returned)
			require.Equal(test, 0, body.reads)
			require.False(test, body.closed)
			require.Equal(test, 1, upstream.calls)
			require.EqualValues(test, 42, upstream.accountID)
			require.Equal(test, 17, upstream.concurrency)
			stored, err := cache.Get(ctx, key).Bytes()
			require.NoError(test, err)
			require.NoError(test, json.Unmarshal(stored, &record))
			require.True(test, record.Invalidated)
			contents, err := io.ReadAll(returned.Body)
			require.NoError(test, err)
			require.Equal(test, "data: hello\n\ndata: [DONE]\n\n", string(contents))
			require.NoError(test, returned.Body.Close())
		})
	}
}
