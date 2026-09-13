//go:build unit

package service

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/openai_compat"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

const correctCandyLine = `CANDY_RESULT=21`

func candyEvents(t *testing.T, body string) []TestEvent {
	t.Helper()
	var events []TestEvent
	for _, line := range strings.Split(body, "\n") {
		if strings.HasPrefix(line, "data: ") {
			var event TestEvent
			require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event))
			events = append(events, event)
		}
	}
	return events
}

func requireCandyVerdict(t *testing.T, body, verdict string) {
	t.Helper()
	var results []CandyTestResult
	for _, event := range candyEvents(t, body) {
		if event.Type == "candy_result" {
			encoded, err := json.Marshal(event.Data)
			require.NoError(t, err)
			var result CandyTestResult
			require.NoError(t, json.Unmarshal(encoded, &result))
			results = append(results, result)
		}
	}
	require.Len(t, results, 1)
	require.Equal(t, verdict, results[0].Verdict)
	require.Equal(t, 21, results[0].Expected)
}

func TestCandyParseFinalAnswer(t *testing.T) {
	for _, text := range []string{correctCandyLine, "分析里出现21、29、35。\r\n" + correctCandyLine + "\r\n\n"} {
		answer, err := parseCandyAnswer(text)
		require.NoError(t, err)
		require.Equal(t, 21, *answer)
	}
	for _, text := range []string{
		"21", "", correctCandyLine + "\n补充说明", correctCandyLine + " trailing",
		"\x60\x60\x60\n" + correctCandyLine + "\n\x60\x60\x60",
		"CANDY_RESULT=21或29", "CANDY_RESULT=21,29,35", "CANDY_RESULT=21颗",
		"CANDY_RESULT=null", "CANDY_RESULT=21.0", "CANDY_RESULT=2.1e1",
		"CANDY_RESULT=+21", "CANDY_RESULT=021", "CANDY_RESULT=99999999999999999999999999",
	} {
		t.Run(text, func(t *testing.T) {
			_, err := parseCandyAnswer(text)
			require.Error(t, err)
		})
	}
}

func TestCandyPromptPreservesOriginalQuestionWithoutAddedHints(t *testing.T) {
	expectedQuestion := `在一个黑色的袋子里放有三种口味的糖果，每种糖果有两种不同的形状（圆形和五角星形，不同的形状靠手感可以分辨）。现已知不同口味的糖和不同形状的数量统计如下表。参赛者需要在活动前决定摸出的糖果数目，那么，最少取出多少个糖果才能保证手中同时拥有不同形状的苹果味和桃子味的糖？（同时手中有圆形苹果味匹配五角星桃子味糖果，或者有圆形桃子味匹配五角星苹果味糖果都满足要求）
苹果味 桃子味 西瓜味
圆形 7 9 8
五角星形 7 6 4`
	question, instructions, found := strings.Cut(candyTestPrompt, "\n\n回答格式要求：")
	require.True(t, found)
	require.Equal(t, expectedQuestion, question)
	require.Equal(t, "可以先分析，但最后必须给出一个唯一的最终答案。最后一行严格使用 CANDY_RESULT=数字，将“数字”替换为你认为正确的最少糖果总数。该行不要包含单位、其它数字、条件或解释，不要使用代码围栏，之后不要再添加文字。", instructions)
	for _, hint := range []string{"按形状", "配额", "21", "29", "35"} {
		require.NotContains(t, candyTestPrompt, hint)
	}
}

func TestCandyEvaluation(t *testing.T) {
	for _, test := range []struct {
		name, output, verdict           string
		completed, successful, overflow bool
	}{
		{"correct", correctCandyLine, "pass", true, true, false},
		{"29 is wrong", "分析提到21\nCANDY_RESULT=29", "incorrect", true, true, false},
		{"35 is wrong", "分析提到21和29\nCANDY_RESULT=35", "incorrect", true, true, false},
		{"just number", "21", "invalid_format", true, true, false},
		{"unfinished", correctCandyLine, "inconclusive", false, true, false},
		{"failed request", correctCandyLine, "inconclusive", true, false, false},
		{"overflow", correctCandyLine, "inconclusive", true, true, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			state := &candyTestState{started: time.Now(), completed: test.completed, overflow: test.overflow}
			state.output.WriteString(test.output)
			result := evaluateCandyTest(state, test.successful)
			require.Equal(t, test.verdict, result.Verdict)
			require.GreaterOrEqual(t, result.DurationMs, int64(0))
		})
	}
}

func TestCandyEventsAreBoundedIsolatedAndEmittedOnce(t *testing.T) {
	service := &AccountTestService{}
	ctx, recorder := newTestContext()
	state := &candyTestState{started: time.Now()}
	ctx.Set(candyTestContextKey, state)
	service.sendEvent(ctx, TestEvent{Type: "content", Text: correctCandyLine[:8]})
	service.sendEvent(ctx, TestEvent{Type: "content", Text: correctCandyLine[8:]})
	markCandyCompletion(ctx, true)
	service.sendEvent(ctx, TestEvent{Type: "test_complete", Success: true})
	service.observeCandyTestEvent(ctx, TestEvent{Type: "error"})
	requireCandyVerdict(t, recorder.Body.String(), "pass")
	events := candyEvents(t, recorder.Body.String())
	require.Equal(t, "candy_result", events[len(events)-2].Type)
	require.Equal(t, "test_complete", events[len(events)-1].Type)

	other, otherRecorder := newTestContext()
	service.sendEvent(other, TestEvent{Type: "content", Text: "hi"})
	service.sendEvent(other, TestEvent{Type: "test_complete", Success: true})
	require.NotContains(t, otherRecorder.Body.String(), "candy_result")

	large, largeRecorder := newTestContext()
	largeState := &candyTestState{started: time.Now(), completed: true}
	large.Set(candyTestContextKey, largeState)
	service.observeCandyTestEvent(large, TestEvent{Type: "content", Text: strings.Repeat("a", maxCandyOutputBytes+1)})
	service.observeCandyTestEvent(large, TestEvent{Type: "content", Text: correctCandyLine})
	service.sendEvent(large, TestEvent{Type: "test_complete", Success: true})
	require.Zero(t, largeState.output.Len())
	requireCandyVerdict(t, largeRecorder.Body.String(), "inconclusive")
}

func TestCandyStreamCompletion(t *testing.T) {
	service := &AccountTestService{}
	encoded, err := json.Marshal(correctCandyLine)
	require.NoError(t, err)
	for _, test := range []struct {
		name, body, verdict string
		process             func(*gin.Context, io.Reader) error
	}{
		{"responses", fmt.Sprintf("data: {\"type\":\"response.output_text.delta\",\"delta\":%s}\n\ndata: {\"type\":\"response.completed\"}\n\n", encoded), "pass", service.processOpenAIStream},
		{"responses incomplete", fmt.Sprintf("data: {\"type\":\"response.output_text.delta\",\"delta\":%s}\n\ndata: {\"type\":\"response.incomplete\"}\n\n", encoded), "inconclusive", service.processOpenAIStream},
		{"responses missing terminal", fmt.Sprintf("data: {\"type\":\"response.output_text.delta\",\"delta\":%s}\n\n", encoded), "inconclusive", service.processOpenAIStream},
		{"chat", fmt.Sprintf("data: {\"choices\":[{\"delta\":{\"content\":%s},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", encoded), "pass", service.processOpenAIChatCompletionsStream},
		{"chat length", fmt.Sprintf("data: {\"choices\":[{\"delta\":{\"content\":%s},\"finish_reason\":\"length\"}]}\n\ndata: [DONE]\n\n", encoded), "inconclusive", service.processOpenAIChatCompletionsStream},
		{"chat no finish", fmt.Sprintf("data: {\"choices\":[{\"delta\":{\"content\":%s}}]}\n\ndata: [DONE]\n\n", encoded), "inconclusive", service.processOpenAIChatCompletionsStream},
		{"claude", fmt.Sprintf("data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":%s}}\n\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\ndata: {\"type\":\"message_stop\"}\n\n", encoded), "pass", service.processClaudeStream},
		{"claude truncated", fmt.Sprintf("data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":%s}}\n\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"}}\n\ndata: {\"type\":\"message_stop\"}\n\n", encoded), "inconclusive", service.processClaudeStream},
		{"claude missing reason", fmt.Sprintf("data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":%s}}\n\n", encoded), "inconclusive", service.processClaudeStream},
		{"gemini ignores thoughts", fmt.Sprintf("data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":%s},{\"text\":\"not a final answer\",\"thought\":true}]},\"finishReason\":\"STOP\"}]}\n\n", encoded), "pass", service.processGeminiStream},
		{"gemini limit", fmt.Sprintf("data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":%s}]},\"finishReason\":\"MAX_TOKENS\"}]}\n\n", encoded), "inconclusive", service.processGeminiStream},
		{"gemini missing reason", fmt.Sprintf("data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":%s}]}}]}\n\n", encoded), "inconclusive", service.processGeminiStream},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, recorder := newTestContext()
			ctx.Set(candyTestContextKey, &candyTestState{started: time.Now()})
			_ = test.process(ctx, strings.NewReader(test.body))
			requireCandyVerdict(t, recorder.Body.String(), test.verdict)
		})
	}
}

func TestCandyRequestUsesFixedPromptAndExistingAccountRouting(t *testing.T) {
	for _, test := range []struct {
		name, accountType string
		responses         bool
	}{
		{"oauth", AccountTypeOAuth, true},
		{"apikey responses", AccountTypeAPIKey, true},
		{"apikey chat", AccountTypeAPIKey, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, recorder := newTestContext()
			account := &Account{ID: 42, Platform: PlatformOpenAI, Type: test.accountType, Concurrency: 1,
				Credentials: map[string]any{"api_key": "sk-test", "access_token": "test-token", "base_url": "https://relay.example.com/v1", "model_mapping": map[string]any{"selected-text": "gpt-5.4"}},
				Extra:       map[string]any{openai_compat.ExtraKeyResponsesSupported: test.responses},
			}
			encoded, err := json.Marshal(correctCandyLine)
			require.NoError(t, err)
			body := fmt.Sprintf("data: {\"type\":\"response.output_text.delta\",\"delta\":%s}\n\ndata: {\"type\":\"response.completed\"}\n\n", encoded)
			if !test.responses {
				body = fmt.Sprintf("data: {\"choices\":[{\"delta\":{\"content\":%s},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", encoded)
			}
			repo := &openAIAccountTestRepo{mockAccountRepoForGemini: mockAccountRepoForGemini{accountsByID: map[int64]*Account{42: account}}}
			upstream := &queuedHTTPUpstream{responses: []*http.Response{newJSONResponse(http.StatusOK, body)}}
			service := &AccountTestService{accountRepo: repo, httpUpstream: upstream, cfg: &config.Config{}}
			require.NoError(t, service.TestAccountConnection(ctx, 42, "selected-text", "client supplied answer", " CANDY "))
			require.Len(t, upstream.requests, 1)
			request := upstream.requests[0]
			payload, err := io.ReadAll(request.Body)
			require.NoError(t, err)
			require.Equal(t, "gpt-5.4", gjson.GetBytes(payload, "model").String())
			promptPath := "input.0.content.0.text"
			if !test.responses {
				promptPath = "messages.0.content"
				require.Equal(t, "/v1/chat/completions", request.URL.Path)
			}
			require.Equal(t, candyTestPrompt, gjson.GetBytes(payload, promptPath).String())
			require.NotContains(t, string(payload), "client supplied answer")
			require.NotContains(t, candyTestPrompt, correctCandyLine)
			deadline, exists := request.Context().Deadline()
			require.True(t, exists)
			require.LessOrEqual(t, time.Until(deadline), 180*time.Second)
			if test.accountType == AccountTypeOAuth {
				require.Equal(t, "Bearer test-token", request.Header.Get("Authorization"))
			} else {
				require.Equal(t, "Bearer sk-test", request.Header.Get("Authorization"))
			}
			requireCandyVerdict(t, recorder.Body.String(), "pass")
			require.Zero(t, repo.setErrorID)
		})
	}
}

func TestCandyRejectsUnsupportedInputsBeforeUpstream(t *testing.T) {
	for _, test := range []struct {
		name, platform, model, mapped string
		options                       AccountTestOptions
	}{
		{name: "unsupported platform", platform: PlatformGrok, model: "grok-text"},
		{name: "image model", platform: PlatformOpenAI, model: "gpt-image-1"},
		{name: "mapped image", platform: PlatformOpenAI, model: "alias", mapped: "gpt-image-1"},
		{name: "media input", platform: PlatformOpenAI, model: "gpt-5.4", options: AccountTestOptions{ImageDataURL: "data:image/png;base64,QUJD"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			account := &Account{ID: 42, Platform: test.platform, Type: AccountTypeAPIKey, Credentials: map[string]any{"model_mapping": map[string]any{test.model: test.mapped}}}
			repo := &openAIAccountTestRepo{mockAccountRepoForGemini: mockAccountRepoForGemini{accountsByID: map[int64]*Account{42: account}}}
			upstream := &queuedHTTPUpstream{}
			service := &AccountTestService{accountRepo: repo, httpUpstream: upstream}
			ctx, recorder := newTestContext()
			require.Error(t, service.TestAccountConnection(ctx, 42, test.model, "", "candy", test.options))
			require.Empty(t, upstream.requests)
			requireCandyVerdict(t, recorder.Body.String(), "inconclusive")
		})
	}
}

func TestCandyPayloadOverrideLeavesNormalTestsUnchanged(t *testing.T) {
	ctx, _ := newTestContext()
	payload := createOpenAITestPayload("gpt-5.4", false)
	applyCandyTestPayload(ctx, payload)
	encoded, err := json.Marshal(payload)
	require.NoError(t, err)
	require.Equal(t, "hi", gjson.GetBytes(encoded, "input.0.content.0.text").String())
	ctx.Set(candyTestContextKey, &candyTestState{})
	claudePayload := map[string]any{"messages": []any{}, "max_tokens": 256, "model": "claude-text"}
	applyCandyTestPayload(ctx, claudePayload)
	encoded, err = json.Marshal(claudePayload)
	require.NoError(t, err)
	require.Equal(t, candyTestPrompt, gjson.GetBytes(encoded, "messages.0.content.0.text").String())
	require.Equal(t, int64(4096), gjson.GetBytes(encoded, "max_tokens").Int())
	require.Equal(t, "claude-text", gjson.GetBytes(encoded, "model").String())
}
