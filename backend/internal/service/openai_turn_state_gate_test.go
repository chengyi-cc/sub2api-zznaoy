package service

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/turnstate"
	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
)

func TestTurnStateGateBlocksBeforeTransportAndAllowsNextReadyAccount(test *testing.T) {
	ctx := context.Background()
	cache := redis.NewClient(&redis.Options{Addr: miniredis.RunT(test).Addr()})
	defer cache.Close()
	runtime := turnstate.NewRuntime(func(config turnstate.Config) (*turnstate.Manager, error) {
		return turnstate.New(config, cache, func(context.Context, int64, http.Header) (http.Header, turnstate.Options, bool) {
			return nil, turnstate.Options{}, false
		})
	})
	defer runtime.Close()
	config := turnstate.Config{ProxyHost: "proxy.example:7778", ProxyUsername: "test_{country}_{session}", ProxyPassword: "test"}
	require.NoError(test, runtime.Update(config, nil))
	upstream := &turnStateResponseUpstream{response: &http.Response{StatusCode: 200, Header: make(http.Header)}}
	gateway := &OpenAIGatewayService{turnStateAuto: runtime, httpUpstream: upstream}
	account := &Account{ID: 42, Platform: PlatformOpenAI, Type: AccountTypeSetupToken, Concurrency: 9, Extra: map[string]any{turnstate.EnabledKey: true}}
	request := func(model string) *http.Request {
		created, err := http.NewRequest(http.MethodPost, "https://chatgpt.com/backend-api/codex/responses", strings.NewReader(`{"model":"`+model+`","stream":true}`))
		require.NoError(test, err)
		return created
	}
	raw := make([]byte, 249)
	raw[0] = 0x80
	binary.BigEndian.PutUint64(raw[1:9], uint64(time.Now().Unix()))
	record, err := turnstate.Parse(base64.URLEncoding.EncodeToString(raw), "gpt-6-astra", time.Now())
	require.NoError(test, err)
	store := func(accountID int64, value turnstate.Record) {
		payload, marshalErr := json.Marshal(value)
		require.NoError(test, marshalErr)
		key := fmt.Sprintf("codex:turn-state:v1:%d:%x", accountID, sha256.Sum256([]byte(value.Model)))
		require.NoError(test, cache.Set(ctx, key, payload, time.Hour).Err())
	}
	for _, state := range []string{"missing", "expired", "invalidated", "wrong-identity", "wrong-profile", "wrong-source"} {
		test.Run(state, func(test *testing.T) {
			candidate := record
			switch state {
			case "expired":
				candidate.ExpiresAt = time.Now().Add(-time.Minute)
			case "invalidated":
				candidate.Invalidated = true
			case "wrong-identity":
				candidate.Identity = "different-account"
			case "wrong-profile":
				candidate.Profile = turnstate.ProfilePro
			case "wrong-source":
				candidate.Source = turnstate.SourceIPv6
			}
			if state != "missing" {
				store(42, candidate)
			}
			sent := request(record.Model)
			sent.Header.Set(turnstate.Header, record.Value)
			response, blockedErr := gateway.doOpenAIUpstream(sent, "", account)
			require.Nil(test, response)
			var failure *UpstreamFailoverError
			require.ErrorAs(test, blockedErr, &failure)
			require.True(test, failure.IsTurnStateUnavailable())
			require.True(test, failure.ShouldRetryNextAccount())
			require.False(test, failure.ShouldReportAccountScheduleFailure())
			require.Equal(test, 503, failure.StatusCode)
			require.Equal(test, 0, upstream.calls)
			require.Same(test, blockedErr, gateway.handleOpenAIUpstreamTransportError(ctx, nil, account, blockedErr, false))
		})
	}
	store(43, record)
	ready := *account
	ready.ID = 43
	sent := request(record.Model)
	_, err = gateway.doOpenAIUpstream(sent, "", &ready)
	require.NoError(test, err)
	require.Equal(test, record.Value, sent.Header.Get(turnstate.Header))
	require.Equal(test, 1, upstream.calls)
	require.EqualValues(test, 43, upstream.accountID)
	require.Equal(test, 9, upstream.concurrency)
	for _, model := range []string{"codex-auto-review", "gpt-5.6-terra", "gpt-5.4"} {
		_, err = gateway.doOpenAIUpstream(request(model), "", account)
		require.NoError(test, err)
	}
	account.Extra[turnstate.EnabledKey] = false
	_, err = gateway.doOpenAIUpstream(request(record.Model), "", account)
	require.NoError(test, err)
	account.Extra[turnstate.EnabledKey] = true
	disabled := false
	config.RequireValidState = &disabled
	require.NoError(test, runtime.Update(config, nil))
	_, err = gateway.doOpenAIUpstream(request(record.Model), "", account)
	require.NoError(test, err)
}

func TestTurnStateGateWithoutSourceAndWebsocketContinuation(test *testing.T) {
	runtime := turnstate.NewRuntime(func(config turnstate.Config) (*turnstate.Manager, error) { return turnstate.New(config, nil, nil) })
	defer runtime.Close()
	require.NoError(test, runtime.Update(turnstate.Config{}, nil))
	gateway := &OpenAIGatewayService{turnStateAuto: runtime}
	account := &Account{Platform: PlatformOpenAI, Type: AccountTypeSetupToken, Extra: map[string]any{turnstate.EnabledKey: true}}
	err := gateway.requireTurnStateAuto(account, "gpt-6-astra", false)
	var failure *UpstreamFailoverError
	require.ErrorAs(test, err, &failure)
	require.True(test, failure.ShouldRetryNextAccount())
	require.NoError(test, gateway.requireTurnStateAuto(account, "gpt-5.6-terra", false))
	err = stopTurnStateFailoverAfterFirstTurn(err, 2)
	require.ErrorAs(test, err, &failure)
	require.False(test, failure.ShouldRetryNextAccount())
	require.False(test, failure.ShouldReportAccountScheduleFailure())
	disabled := false
	require.NoError(test, runtime.Update(turnstate.Config{RequireValidState: &disabled}, nil))
	require.NoError(test, gateway.requireTurnStateAuto(account, "gpt-6-astra", false))
}
