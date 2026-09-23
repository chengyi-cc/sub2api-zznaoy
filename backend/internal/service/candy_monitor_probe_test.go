//go:build unit

package service

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/openai_compat"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestCandyMonitorBackgroundUsesPuzzleAndPreservesAccountHealth(t *testing.T) {
	for _, tc := range []struct {
		name          string
		status        int
		body, verdict string
	}{
		{"21", 200, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"CANDY_RESULT=21\"}\n\ndata: {\"type\":\"response.completed\"}\n\n", "pass"},
		{"29", 200, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"CANDY_RESULT=29\"}\n\ndata: {\"type\":\"response.completed\"}\n\n", "incorrect"},
		{"unauthorized", 401, `{"error":{"message":"bad key"}}`, "inconclusive"},
		{"rate limited", 429, `{"error":{"type":"usage_limit_reached","resets_at":2000000000}}`, "inconclusive"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := &Account{ID: 42, Platform: PlatformOpenAI, Type: AccountTypeAPIKey, Credentials: map[string]any{"api_key": "sk-test", "base_url": "https://relay.example.com/v1"}, Extra: map[string]any{openai_compat.ExtraKeyResponsesSupported: true}}
			repo := &openAIAccountTestRepo{mockAccountRepoForGemini: mockAccountRepoForGemini{accountsByID: map[int64]*Account{42: a}}}
			response := newJSONResponse(tc.status, tc.body)
			response.Header.Set("x-ratelimit-reset-requests", "30s")
			upstream := &queuedHTTPUpstream{responses: []*http.Response{response}}
			svc := &AccountTestService{accountRepo: repo, httpUpstream: upstream, cfg: &config.Config{}}
			result, text, err := svc.RunCandyTestBackground(context.Background(), 42, CandyMonitorDefaultModel)
			if tc.status == 200 {
				require.NoError(t, err)
				require.Contains(t, text, "CANDY_RESULT=")
			} else {
				require.Error(t, err)
			}
			require.Equal(t, tc.verdict, result.Verdict)
			require.Zero(t, repo.setErrorID)
			require.Zero(t, repo.rateLimitedID)
			require.Zero(t, repo.clearedErrorID)
			require.Len(t, upstream.requests, 1)
			body, err := io.ReadAll(upstream.requests[0].Body)
			require.NoError(t, err)
			require.Equal(t, candyTestPrompt, gjson.GetBytes(body, "input.0.content.0.text").String())
			require.Equal(t, CandyMonitorDefaultModel, gjson.GetBytes(body, "model").String(), fmt.Sprint(tc))
		})
	}
}
