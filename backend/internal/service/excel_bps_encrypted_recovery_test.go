package service

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func bpsEncryptedRejection() *http.Response {
	return &http.Response{StatusCode: 400, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(
		`{"error":{"code":"invalid_encrypted_content","message":"The encrypted content fe12...3a-0 could not be verified. Reason: Encrypted content could not be decrypted or parsed."}}`))}
}

func bpsRecoverySuccess() *http.Response {
	return excelBPSTestResponse("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_recovered\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n")
}

func bpsRecoveryContext(body []byte, key int64, thread string) (*gin.Context, *httptest.ResponseRecorder) {
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("session_id", thread)
	c.Set("api_key", &APIKey{ID: key})
	return c, rec
}

const bpsRecoveryBody = `{"model":"gpt-6-astra","input":[{"type":"message","role":"user","content":"Keep all my instructions"},{"type":"reasoning","encrypted_content":"fe12-bad-cipher-3a-0"},{"type":"reasoning","encrypted_content":"other-valid-cipher"},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Recorded answer"}]},{"type":"message","role":"user","content":"Continue"}],"tools":[{"type":"custom","name":"exec"}]}`

func TestExcelBPSInvalidReasoningRecoveryAndScopedReplay(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprint(stream), func(t *testing.T) {
			body := []byte(strings.Replace(bpsRecoveryBody, `"model":`, fmt.Sprintf(`"stream":%v,"model":`, stream), 1))
			up := &httpUpstreamRecorder{responses: []*http.Response{bpsEncryptedRejection(), bpsRecoverySuccess(), bpsRecoverySuccess()}}
			svc := openAIClientToolsTestService(up)
			c, rec := bpsRecoveryContext(body, 91, "recovery-thread")
			result, err := svc.Forward(context.Background(), c, excelAccount(), body)
			require.NoError(t, err)
			require.NotNil(t, result)
			require.Equal(t, 200, rec.Code)
			require.Len(t, up.requests, 2)
			require.Contains(t, string(up.bodies[0]), "fe12-bad-cipher-3a-0")
			require.NotContains(t, string(up.bodies[1]), "fe12-bad-cipher-3a-0")
			require.Contains(t, string(up.bodies[1]), "other-valid-cipher")
			for _, want := range []string{"Keep all my instructions", "Recorded answer", "Continue"} {
				require.Contains(t, string(up.bodies[1]), want)
			}
			require.Equal(t, up.requests[0].Header.Get("Authorization"), up.requests[1].Header.Get("Authorization"))
			c, rec = bpsRecoveryContext(body, 91, "recovery-thread")
			_, err = svc.Forward(context.Background(), c, excelAccount(), body)
			require.NoError(t, err)
			require.Len(t, up.requests, 3, "verified invalid digest should not incur another rejected request")
			require.Equal(t, up.bodies[1], up.bodies[2], "subsequent history prefix must remain stable")
			for _, boundary := range []string{"key", "thread", "account", "owner", "model"} {
				account, key, thread := excelAccount(), int64(91), "recovery-thread"
				nextBody := body
				switch boundary {
				case "key":
					key = 92
				case "thread":
					thread = "other-thread"
				case "account":
					account.ID++
				case "owner":
					account.Credentials["chatgpt_account_id"] = "other-owner"
				case "model":
					nextBody = []byte(strings.Replace(string(body), "gpt-6-astra", "gpt-5.6-sol", 1))
				}
				before := len(up.requests)
				up.responses = append(up.responses, bpsEncryptedRejection(), bpsRecoverySuccess())
				c, _ = bpsRecoveryContext(nextBody, key, thread)
				_, err = svc.Forward(context.Background(), c, account, nextBody)
				require.NoError(t, err, boundary)
				require.Len(t, up.requests, before+2, boundary+" must not share rejected history state")
				require.Contains(t, string(up.bodies[before]), "fe12-bad-cipher-3a-0", boundary)
			}
		})
	}
}

func TestExcelBPSReasoningRecoveryPreservesToolPayload(t *testing.T) {
	body := []byte(`{"model":"gpt-6-astra","metadata":{"unchanged":true},"input":[{"role":"user","content":"Keep 中文 and all tool results"},{"type":"reasoning","encrypted_content":"fe12-bad-cipher-3a-0"},{"type":"function_call","name":"run_officejs","call_id":"keep","arguments":"{\"amount\":9007199254740993,\"text\":\"\\n\\t\"}"},{"type":"function_call_output","call_id":"keep","output":[{"type":"input_text","text":"result"},{"type":"input_image","file_id":"file-safe","detail":"original"}]},{"type":"reasoning","encrypted_content":"fresh-cipher"}]}`)
	rejection := []byte(`{"error":{"code":"invalid_encrypted_content","param":"input[1].encrypted_content"}}`)
	before := append([]byte(nil), body...)
	next, digests := excelBPSRejectedReasoningRetry(body, rejection)
	require.Len(t, digests, 1)
	require.Equal(t, before, body, "caller request must remain immutable")
	oldItems, newItems := gjson.GetBytes(body, "input").Array(), gjson.GetBytes(next, "input").Array()
	require.Len(t, newItems, len(oldItems)-1)
	for i, oldIndex := range []int{0, 2, 3, 4} {
		require.JSONEq(t, oldItems[oldIndex].Raw, newItems[i].Raw)
	}
	require.Contains(t, string(next), "9007199254740993")
	require.Equal(t, gjson.GetBytes(body, "metadata").Raw, gjson.GetBytes(next, "metadata").Raw)
	for _, invalidError := range []string{
		`{"error":{"code":"invalid_encrypted_content","param":"input[99].encrypted_content"}}`,
		`{"error":{"code":"invalid_encrypted_content","param":"input[2].encrypted_content"}}`,
		`{"error":{"code":"invalid_encrypted_content","param":"input[1].encrypted_content","message":"The encrypted content wrong-cipher could not be verified"}}`,
		`{"error":{"code":"invalid_encrypted_content"}}`,
		`{"error":{"code":"rate_limit_exceeded","param":"input[1].encrypted_content"}}`,
		`{"error":{"code":"invalid_encrypted_content","param":"input[1].encrypted_content"}}trailing`,
	} {
		unchanged, rejected := excelBPSRejectedReasoningRetry(body, []byte(invalidError))
		require.Empty(t, rejected, invalidError)
		require.Equal(t, body, unchanged)
	}
}

func TestExcelBPSReasoningRecoveryRejectsOtherEncryptedContent(t *testing.T) {
	for _, extra := range []string{
		`{"type":"agent_message","content":[{"type":"encrypted_content","encrypted_content":"opaque-agent-task"}]}`,
		`{"type":"message","role":"user","content":[{"type":"encrypted_content","encrypted_content":"opaque-user-task"}]}`,
		`{"type":"compaction_summary","encrypted_content":"only-copy-of-history"}`,
		`{"type":"item_reference","id":"external-history"}`,
	} {
		body := []byte(`{"input":[{"role":"user","content":"Continue"},{"type":"reasoning","encrypted_content":"fe12-bad-cipher-3a-0"},` + extra + "]}")
		rejection, _ := io.ReadAll(bpsEncryptedRejection().Body)
		next, digests := excelBPSRejectedReasoningRetry(body, rejection)
		require.Empty(t, digests, extra)
		require.Equal(t, body, next)
		cached := map[string]struct{}{openAIEncryptedContentDigest("fe12-bad-cipher-3a-0"): {}}
		next, count := excelBPSDropRejectedReasoning(body, cached)
		require.Zero(t, count, "cached recovery must use the same safety checks")
		require.Equal(t, body, next)
	}
}

func TestExcelBPSReasoningRecoveryDoesNotCacheFailedStream(t *testing.T) {
	failed := excelBPSTestResponse("data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"code\":\"server_error\"}}}\n\n")
	up := &httpUpstreamRecorder{responses: []*http.Response{bpsEncryptedRejection(), failed, bpsEncryptedRejection(), bpsRecoverySuccess()}}
	svc := openAIClientToolsTestService(up)
	c, _ := bpsRecoveryContext([]byte(bpsRecoveryBody), 91, "failed-stream")
	_, err := svc.Forward(context.Background(), c, excelAccount(), []byte(bpsRecoveryBody))
	require.Error(t, err)
	c, _ = bpsRecoveryContext([]byte(bpsRecoveryBody), 91, "failed-stream")
	_, err = svc.Forward(context.Background(), c, excelAccount(), []byte(bpsRecoveryBody))
	require.NoError(t, err)
	require.Len(t, up.requests, 4)
	require.Contains(t, string(up.bodies[2]), "fe12-bad-cipher-3a-0")
}

func TestExcelBPSEncryptedHistoryRecoveryFailsClosed(t *testing.T) {
	for name, input := range map[string]string{
		"compacted history":              `[{"type":"compaction","encrypted_content":"fe12-bad-cipher-3a-0"},{"role":"user","content":"Continue"}]`,
		"compaction alongside reasoning": `[{"type":"compaction","encrypted_content":"keep-summary"},{"type":"reasoning","encrypted_content":"fe12-bad-cipher-3a-0"},{"role":"user","content":"Continue"}]`,
		"unidentified cipher":            `[{"type":"reasoning","encrypted_content":"different-cipher"},{"role":"user","content":"Continue"}]`,
		"ambiguous cipher prefix":        `[{"type":"reasoning","encrypted_content":"fe12-one-3a-0"},{"type":"reasoning","encrypted_content":"fe12-two-3a-0"},{"role":"user","content":"Continue"}]`,
		"no plaintext user history":      `[{"type":"reasoning","encrypted_content":"fe12-bad-cipher-3a-0"}]`,
	} {
		t.Run(name, func(t *testing.T) {
			up := &httpUpstreamRecorder{responses: []*http.Response{bpsEncryptedRejection()}}
			svc := openAIClientToolsTestService(up)
			body := []byte(`{"model":"gpt-6-astra","input":` + input + "}")
			c, rec := bpsRecoveryContext(body, 91, "unsafe-recovery")
			_, err := svc.Forward(context.Background(), c, excelAccount(), body)
			require.Error(t, err)
			require.Len(t, up.requests, 1)
			require.Equal(t, "invalid_encrypted_content", gjson.Get(rec.Body.String(), "error.code").String())
			require.NotContains(t, rec.Body.String(), "fe12")
		})
	}
}

func TestExcelBPSInvalidReasoningRetryBoundAndFailureNotCached(t *testing.T) {
	up := &httpUpstreamRecorder{responses: []*http.Response{bpsEncryptedRejection(), bpsEncryptedRejection(), bpsEncryptedRejection(), bpsEncryptedRejection()}}
	svc := openAIClientToolsTestService(up)
	for round := 1; round <= 2; round++ {
		c, rec := bpsRecoveryContext([]byte(bpsRecoveryBody), 91, "failed-recovery")
		_, err := svc.Forward(context.Background(), c, excelAccount(), []byte(bpsRecoveryBody))
		require.Error(t, err)
		require.Len(t, up.requests, round*2)
		require.Equal(t, "invalid_encrypted_content", gjson.Get(rec.Body.String(), "error.code").String())
		require.Contains(t, string(up.bodies[(round-1)*2]), "fe12-bad-cipher-3a-0")
	}
}
