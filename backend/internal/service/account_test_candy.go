package service

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

const AccountTestModeCandy = "candy"
const candyTestContextKey = "account_test_candy"
const candyAnswerPrefix = "CANDY_RESULT="
const maxCandyOutputBytes = 64 << 10

const candyTestPrompt = `在一个黑色的袋子里放有三种口味的糖果，每种糖果有两种不同的形状（圆形和五角星形，不同的形状靠手感可以分辨）。现已知不同口味的糖和不同形状的数量统计如下表。参赛者需要在活动前决定摸出的糖果数目，那么，最少取出多少个糖果才能保证手中同时拥有不同形状的苹果味和桃子味的糖？（同时手中有圆形苹果味匹配五角星桃子味糖果，或者有圆形桃子味匹配五角星苹果味糖果都满足要求）
苹果味 桃子味 西瓜味
圆形 7 9 8
五角星形 7 6 4

回答格式要求：可以先分析，但最后必须给出一个唯一的最终答案。最后一行严格使用 CANDY_RESULT=数字，将“数字”替换为你认为正确的最少糖果总数。该行不要包含单位、其它数字、条件或解释，不要使用代码围栏，之后不要再添加文字。`

type CandyTestResult struct {
	CaseID     string `json:"case_id"`
	Verdict    string `json:"verdict"`
	Reason     string `json:"reason"`
	Expected   int    `json:"expected"`
	Actual     *int   `json:"actual,omitempty"`
	DurationMs int64  `json:"duration_ms"`
}

type candyTestState struct {
	output    strings.Builder
	started   time.Time
	completed bool
	overflow  bool
	graded    bool
}

func candyState(c *gin.Context) *candyTestState {
	value, _ := c.Get(candyTestContextKey)
	state, _ := value.(*candyTestState)
	return state
}

func validateCandyTest(account *Account, modelID string, opts AccountTestOptions) error {
	if account.IsSyntheticUITest() || (account.Platform != PlatformOpenAI && account.Platform != PlatformAnthropic && account.Platform != PlatformGemini) {
		return fmt.Errorf("Candy test supports OpenAI, Anthropic and Gemini text accounts only")
	}
	if opts.ImageDataURL != "" || opts.AudioDataURL != "" {
		return fmt.Errorf("Candy test does not accept image or audio inputs")
	}
	for _, model := range []string{modelID, account.GetMappedModel(modelID)} {
		lower := strings.ToLower(model)
		for _, marker := range []string{"image", "imagine", "video", "audio", "realtime", "tts", "whisper", "embedding", "moderation"} {
			if strings.Contains(lower, marker) {
				return fmt.Errorf("Candy test requires a text model, not %s", model)
			}
		}
	}
	return nil
}

func applyCandyTestPayload(c *gin.Context, payload map[string]any) {
	if candyState(c) == nil {
		return
	}
	if _, exists := payload["input"]; exists {
		payload["input"] = []map[string]any{{"role": "user", "content": []map[string]any{{"type": "input_text", "text": candyTestPrompt}}}}
	}
	if _, exists := payload["messages"]; exists {
		payload["messages"] = []map[string]any{{"role": "user", "content": []map[string]any{{"type": "text", "text": candyTestPrompt}}}}
	}
	if _, exists := payload["max_tokens"]; exists {
		payload["max_tokens"] = 4096
	}
}

func markCandyCompletion(c *gin.Context, complete bool) {
	if state := candyState(c); state != nil {
		state.completed = complete
	}
}

func parseCandyAnswer(text string) (*int, error) {
	text = strings.TrimSpace(text)
	if index := strings.LastIndexByte(text, '\n'); index >= 0 {
		text = strings.TrimSpace(text[index+1:])
	}
	if !strings.HasPrefix(text, candyAnswerPrefix) {
		return nil, fmt.Errorf("missing final answer line")
	}
	number := strings.TrimPrefix(text, candyAnswerPrefix)
	answer, err := strconv.Atoi(number)
	if err != nil || strconv.Itoa(answer) != number {
		return nil, fmt.Errorf("final answer must be a single integer")
	}
	return &answer, nil
}

func evaluateCandyTest(state *candyTestState, successful bool) CandyTestResult {
	result := CandyTestResult{
		CaseID: "candy-shape-v1", Verdict: "inconclusive", Reason: "incomplete",
		Expected:   21,
		DurationMs: time.Since(state.started).Milliseconds(),
	}
	if !successful || !state.completed {
		return result
	}
	if state.overflow {
		result.Reason = "output_limit"
		return result
	}
	answer, err := parseCandyAnswer(state.output.String())
	if err != nil {
		result.Verdict, result.Reason = "invalid_format", "format"
		return result
	}
	result.Actual = answer
	result.Verdict, result.Reason = "incorrect", "answer"
	if *answer == result.Expected {
		result.Verdict, result.Reason = "pass", "correct"
	}
	return result
}

func (s *AccountTestService) observeCandyTestEvent(c *gin.Context, event TestEvent) {
	state := candyState(c)
	if state == nil || state.graded {
		return
	}
	if event.Type == "content" {
		if len(event.Text) > maxCandyOutputBytes-state.output.Len() {
			state.overflow = true
		} else if !state.overflow {
			state.output.WriteString(event.Text)
		}
	}
	if event.Type == "test_complete" || event.Type == "error" {
		state.graded = true
		s.sendEvent(c, TestEvent{Type: "candy_result", Data: evaluateCandyTest(state, event.Type == "test_complete" && event.Success)})
	}
}
