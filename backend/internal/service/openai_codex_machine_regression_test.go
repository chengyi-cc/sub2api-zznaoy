package service

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	coderws "github.com/coder/websocket"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestCodexMachineReferenceVectors(t *testing.T) {
	ids := &codexFingerprintIDs{machineSeed: "test-seed", mode: codexFingerprintMachine}
	require.Equal(t, "826a9879-9767-4dda-8a9a-80fed93c0992", ids.machinePseudonym(" session-one "))
	original := "019539e8-8c00-7000-8000-000000000001"
	mapped := ids.machinePseudonym(original)
	require.Equal(t, "019539e8-8c00-7294-8fda-cfb7b8788254", mapped)
	require.Equal(t, mapped, ids.machinePseudonym(mapped))
	for _, suffix := range []string{"", ":0", ":007", ":editor", ":"} {
		require.Equal(t, mapped+suffix, ids.machineWindowPseudonym(" "+original+suffix+" "))
	}
	for _, raw := range []string{"", "thread-one:0", "non-uuid", " non-uuid:03 "} {
		require.Equal(t, raw, ids.machineWindowPseudonym(raw))
	}
}

func TestCodexMachineMetadataPreservesShape(t *testing.T) {
	ids := &codexFingerprintIDs{machineSeed: "test-seed", mode: codexFingerprintMachine, installationID: "device", machineSandboxTag: "seatbelt"}
	raw := `{"session_id":"first","session_id":"last","parent_thread_id":"parent","forked_from_thread_id":"fork","turn_id":"unchanged","sandbox":"seccomp","unknown":9007199254740993,"nested":{"value":9007199254740995}}`
	rewritten := rewriteCodexMachineTurnMetadata(raw, ids)
	require.Equal(t, 1, strings.Count(rewritten, `"session_id"`))
	require.Equal(t, ids.machinePseudonym("last"), gjson.Get(rewritten, "session_id").String())
	require.Equal(t, ids.machinePseudonym("parent"), gjson.Get(rewritten, "parent_thread_id").String())
	require.Equal(t, ids.machinePseudonym("fork"), gjson.Get(rewritten, "forked_from_thread_id").String())
	require.Equal(t, "unchanged", gjson.Get(rewritten, "turn_id").String())
	require.Equal(t, "seatbelt", gjson.Get(rewritten, "sandbox").String())
	require.Equal(t, "9007199254740993", gjson.Get(rewritten, "unknown").Raw)
	require.Equal(t, "9007199254740995", gjson.Get(rewritten, "nested.value").Raw)
	for _, raw := range []string{`{"sandbox":"none"}`, `{"sandbox":false}`, `{"unknown":1}`, `[]`, `invalid`} {
		require.Equal(t, raw, rewriteCodexMachineTurnMetadata(raw, ids))
	}
	for _, raw := range []string{"{}", `{"client_metadata":null}`, `{"client_metadata":[]}`} {
		next, changed, err := applyCodexFingerprintClientMetadataRaw([]byte(raw), ids)
		require.NoError(t, err)
		require.False(t, changed)
		require.Equal(t, raw, string(next))
	}
	next, _, err := applyCodexFingerprintClientMetadataRaw([]byte(`{"client_metadata":{"session_id":"session-one","turn_id":"keep","large":9007199254740993}}`), ids)
	require.NoError(t, err)
	require.Equal(t, "9007199254740993", gjson.GetBytes(next, "client_metadata.large").Raw)
	require.Equal(t, "keep", gjson.GetBytes(next, "client_metadata.turn_id").String())
	headers := http.Header{"Session_id": {"legacy"}, "Conversation_id": {"drop"}}
	applyCodexMachineHeaders(headers, ids)
	require.Equal(t, "legacy", headers.Get("session_id"))
	require.Empty(t, headers.Get("conversation_id"))
	require.Empty(t, headers.Get("x-codex-installation-id"))
	require.False(t, applyCodexMachineClientMetadata(map[string]any{}, ids))
}

func TestCodexMachineSandboxMatrix(t *testing.T) {
	for _, test := range []struct{ ua, expected string }{
		{"codex_cli_rs/0.146.0 (Mac OS 15.0; arm64) Terminal/1", "seatbelt"},
		{"codex_cli_rs/0.146.0 (Windows 11; x86_64) Terminal/1", "windows_sandbox"},
		{"codex_cli_rs/0.146.0 (Linux 6.8; x86_64) Terminal/1", "seccomp"},
	} {
		ids := &codexFingerprintIDs{mode: codexFingerprintMachine}
		stampCodexMachineSandboxTag(ids, test.ua)
		require.Equal(t, test.expected, ids.machineSandboxTag)
		for _, original := range []string{"seccomp", "seatbelt", "windows_sandbox", "windows_elevated"} {
			expected := test.expected
			if expected == "windows_sandbox" && strings.HasPrefix(original, "windows_") {
				expected = original
			}
			require.Equal(t, expected, rewriteCodexMachineSandbox(original, ids.machineSandboxTag))
		}
		require.Equal(t, "workspace-write", rewriteCodexMachineSandbox("workspace-write", ids.machineSandboxTag))
	}
}

func TestCodexMachineCompactForwardAndProbe(t *testing.T) {
	for _, passthrough := range []bool{false, true} {
		account := newTestOAuthAccount(1, map[string]any{codexFingerprintModeExtraKey: "machine", "openai_oauth_passthrough": passthrough})
		account.Credentials = map[string]any{"access_token": "token", "chatgpt_account_id": "account"}
		requestContext := machineTestContext(10)
		requestContext.Request.URL.Path = "/v1/responses/compact"
		upstream := &httpUpstreamRecorder{err: errors.New("capture only")}
		gateway := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream}
		_, err := gateway.Forward(requestContext.Request.Context(), requestContext, account, []byte(`{"model":"gpt-5.4","input":[],"prompt_cache_key":"session-one","client_metadata":{"session_id":"session-one"}}`))
		require.Error(t, err)
		require.NotNil(t, upstream.lastReq)
		require.Equal(t, upstream.lastReq.Header.Get("session-id"), gjson.GetBytes(upstream.lastBody, "client_metadata.session_id").String())
		require.Equal(t, upstream.lastReq.Header.Get("session-id"), gjson.GetBytes(upstream.lastBody, "prompt_cache_key").String())
		require.NotEqual(t, "session-one", upstream.lastReq.Header.Get("session-id"))
	}
	account := newTestOAuthAccount(1, map[string]any{codexFingerprintModeExtraKey: "machine"})
	account.Credentials = map[string]any{"access_token": "token"}
	upstream := &httpUpstreamRecorder{resp: &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(compactProbeSSESuccessBody))}}
	tester := &AccountTestService{httpUpstream: upstream}
	require.NoError(t, tester.testOpenAICompactConnection(machineTestContext(10), account, "gpt-5.4"))
	req := upstream.lastReq
	metadata := gjson.GetBytes(upstream.lastBody, "client_metadata")
	require.Equal(t, metadata.Get("session_id").String(), req.Header.Get("session-id"))
	require.Equal(t, metadata.Get("thread_id").String(), req.Header.Get("thread-id"))
	require.Empty(t, req.Header.Get("session_id"))
	require.Empty(t, req.Header.Get("conversation_id"))
	require.JSONEq(t, metadata.Get("x-codex-turn-metadata").String(), req.Header.Get("x-codex-turn-metadata"))
	require.Equal(t, metadata.Get("thread_id").String()+":0", metadata.Get("x-codex-window-id").String())
	require.Equal(t, uuid.Version(7), uuid.MustParse(metadata.Get("turn_id").String()).Version())
	turn := gjson.Parse(metadata.Get("x-codex-turn-metadata").String())
	require.Equal(t, "responses_compaction_v2", turn.Get("compaction.implementation").String())
	require.Equal(t, "/root", turn.Get("agent_name").String())
	require.False(t, turn.Get("node_repl_disabled").Bool())
	require.Equal(t, int64(len(upstream.lastBody)), req.ContentLength)
	replay, err := req.GetBody()
	require.NoError(t, err)
	defer replay.Close()
	replayBody, err := io.ReadAll(replay)
	require.NoError(t, err)
	require.Equal(t, upstream.lastBody, replayBody)
}

func TestCodexMachineCompatAndFailover(t *testing.T) {
	for _, messages := range []bool{false, true} {
		account := newTestOAuthAccount(1, map[string]any{codexFingerprintModeExtraKey: "machine"})
		account.Credentials = map[string]any{"access_token": "token"}
		requestContext := machineTestContext(10)
		upstream := &httpUpstreamRecorder{err: errors.New("capture only")}
		gateway := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream}
		body := []byte(`{"model":"gpt-5.4","max_tokens":10,"messages":[{"role":"user","content":"hello"}]}`)
		var err error
		if messages {
			_, err = gateway.ForwardAsAnthropic(requestContext.Request.Context(), requestContext, account, body, "cache", "")
		} else {
			_, err = gateway.forwardAsChatCompletions(requestContext.Request.Context(), requestContext, account, body, "cache", "", false)
		}
		require.Error(t, err)
		require.NotNil(t, upstream.lastReq)
		ids := stagedCodexFingerprintIDs(requestContext, account)
		require.NotNil(t, ids)
		require.Equal(t, ids.machinePseudonym("session-one"), upstream.lastReq.Header.Get("session-id"))
		require.Empty(t, upstream.lastReq.Header.Get("session_id"))
		require.Empty(t, upstream.lastReq.Header.Get("conversation_id"))
		require.False(t, gjson.GetBytes(upstream.lastBody, "client_metadata.x-codex-installation-id").Exists())
		account.Extra[codexFingerprintModeExtraKey] = "off"
		gateway.stageCodexMachineFingerprintIDs(requestContext, account)
		require.Nil(t, stagedCodexFingerprintIDs(requestContext, account))
	}
}

func TestCodexMachineConcurrentMapping(t *testing.T) {
	ids := &codexFingerprintIDs{machineSeed: "test-seed", mode: codexFingerprintMachine}
	var workers sync.WaitGroup
	for index := 0; index < 16; index++ {
		workers.Go(func() {
			for repeat := 0; repeat < 100; repeat++ {
				require.Equal(t, "826a9879-9767-4dda-8a9a-80fed93c0992", ids.machinePseudonym("session-one"))
			}
		})
	}
	workers.Wait()
}

type machineWSConn struct {
	*openAIWSCaptureConn
	responses chan []byte
	done      chan struct{}
	closeOnce sync.Once
}

func (connection *machineWSConn) WriteJSON(ctx context.Context, value any) error {
	if err := connection.openAIWSCaptureConn.WriteJSON(ctx, value); err != nil {
		return err
	}
	select {
	case connection.responses <- []byte(`{"type":"response.completed","response":{"id":"resp_machine","model":"gpt-5.4","usage":{"input_tokens":1,"output_tokens":1}}}`):
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
func (connection *machineWSConn) WriteFrame(ctx context.Context, kind coderws.MessageType, payload []byte) error {
	return connection.WriteJSON(ctx, json.RawMessage(payload))
}
func (connection *machineWSConn) ReadMessage(ctx context.Context) ([]byte, error) {
	select {
	case payload := <-connection.responses:
		return payload, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-connection.done:
		return nil, io.EOF
	}
}
func (connection *machineWSConn) ReadFrame(ctx context.Context) (coderws.MessageType, []byte, error) {
	payload, err := connection.ReadMessage(ctx)
	return coderws.MessageText, payload, err
}
func (connection *machineWSConn) Close() error {
	connection.closeOnce.Do(func() { close(connection.done) })
	return connection.openAIWSCaptureConn.Close()
}

type machineWSDialer struct {
	connection *machineWSConn
	headers    chan http.Header
}

func (dialer *machineWSDialer) Dial(ctx context.Context, wsURL string, headers http.Header, proxyURL string) (openAIWSClientConn, int, http.Header, error) {
	dialer.headers <- headers.Clone()
	return dialer.connection, 0, http.Header{}, nil
}

func TestCodexMachineDirectWebSocketMultipleTurns(t *testing.T) {
	for _, mode := range []string{OpenAIWSIngressModeCtxPool, OpenAIWSIngressModePassthrough} {
		t.Run(mode, func(t *testing.T) {
			cfg := &config.Config{}
			cfg.Gateway.OpenAIWS.Enabled = true
			cfg.Gateway.OpenAIWS.OAuthEnabled = true
			cfg.Gateway.OpenAIWS.ResponsesWebsocketsV2 = true
			cfg.Gateway.OpenAIWS.ModeRouterV2Enabled = true
			cfg.Gateway.OpenAIWS.IngressModeDefault = OpenAIWSIngressModeCtxPool
			cfg.Gateway.OpenAIWS.MaxConnsPerAccount = 1
			cfg.Gateway.OpenAIWS.MaxIdlePerAccount = 1
			cfg.Gateway.OpenAIWS.QueueLimitPerConn = 8
			cfg.Gateway.OpenAIWS.DialTimeoutSeconds = 3
			cfg.Gateway.OpenAIWS.ReadTimeoutSeconds = 3
			cfg.Gateway.OpenAIWS.WriteTimeoutSeconds = 3
			captured := &machineWSConn{openAIWSCaptureConn: &openAIWSCaptureConn{}, responses: make(chan []byte, 8), done: make(chan struct{})}
			dialer := &machineWSDialer{connection: captured, headers: make(chan http.Header, 4)}
			pool := newOpenAIWSConnPool(cfg)
			pool.setClientDialerForTest(dialer)
			gateway := &OpenAIGatewayService{cfg: cfg, httpUpstream: &httpUpstreamRecorder{}, cache: &stubGatewayCache{}, openaiWSResolver: NewOpenAIWSProtocolResolver(cfg), toolCorrector: NewCodexToolCorrector(), openaiWSPool: pool, openaiWSPassthroughDialer: dialer}
			account := newTestOAuthAccount(1, map[string]any{codexFingerprintModeExtraKey: "machine", "openai_oauth_responses_websockets_v2_mode": mode})
			account.Concurrency = 1
			account.Credentials = map[string]any{"access_token": "token"}
			sessionID := "019539e8-8c00-7000-8000-000000000001"
			originalHeaders := http.Header{}
			originalHeaders.Set("session-id", sessionID)
			originalHeaders.Set("thread-id", sessionID)
			originalHeaders.Set("x-codex-parent-thread-id", sessionID)
			originalHeaders.Set("x-codex-window-id", sessionID+":007")
			originalHeaders.Set("x-codex-turn-metadata", `{"session_id":"`+sessionID+`","sandbox":"windows_elevated"}`)
			finished := make(chan error, 1)
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				connection, err := coderws.Accept(writer, request, nil)
				if err != nil {
					finished <- err
					return
				}
				defer connection.CloseNow()
				requestContext, _ := gin.CreateTestContext(httptest.NewRecorder())
				requestContext.Request = request.Clone(request.Context())
				requestContext.Request.Header = originalHeaders.Clone()
				_, first, err := connection.Read(request.Context())
				if err != nil {
					finished <- err
					return
				}
				finished <- gateway.ProxyResponsesWebSocketFromClient(request.Context(), requestContext, connection, account, "token", first, nil)
			}))
			defer server.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
			defer cancel()
			client, _, err := coderws.Dial(ctx, "ws"+strings.TrimPrefix(server.URL, "http"), nil)
			require.NoError(t, err)
			defer client.CloseNow()
			for _, turnID := range []string{"turn-one", "turn-two"} {
				body := map[string]any{"type": "response.create", "model": "gpt-5.4", "store": false, "input": []any{}, "prompt_cache_key": sessionID, "client_metadata": map[string]any{"session_id": sessionID, "thread_id": sessionID, "turn_id": turnID, "x-codex-window-id": sessionID + ":007"}}
				payload, err := json.Marshal(body)
				require.NoError(t, err)
				require.NoError(t, client.Write(ctx, coderws.MessageText, payload))
				_, event, err := client.Read(ctx)
				if err != nil {
					select {
					case serverErr := <-finished:
						t.Fatalf("client: %v; ingress: %v", err, serverErr)
					case <-ctx.Done():
						t.Fatal(err)
					}
				}
				require.NoError(t, err)
				require.Contains(t, string(event), "response.completed")
			}
			client.CloseNow()
			select {
			case <-finished:
			case <-ctx.Done():
				t.Fatal("websocket ingress did not finish")
			}
			var headers http.Header
			select {
			case headers = <-dialer.headers:
			default:
				t.Fatal("no upstream handshake captured")
			}
			expected := resolveCodexFingerprintIDsFromRequest(account, originalHeaders).machinePseudonym(sessionID)
			require.Equal(t, expected, headers.Get("session-id"))
			require.Equal(t, expected, headers.Get("x-codex-parent-thread-id"))
			require.Equal(t, expected+":007", headers.Get("x-codex-window-id"))
			require.Empty(t, headers.Get("session_id"))
			require.Empty(t, headers.Get("conversation_id"))
			captured.mu.Lock()
			writes := append([]map[string]any(nil), captured.writes...)
			captured.mu.Unlock()
			require.Len(t, writes, 2)
			for index, body := range writes {
				metadata := body["client_metadata"].(map[string]any)
				require.Equal(t, expected, metadata["session_id"])
				require.Equal(t, expected, body["prompt_cache_key"])
				require.Equal(t, expected+":007", metadata["x-codex-window-id"])
				require.Equal(t, []string{"turn-one", "turn-two"}[index], metadata["turn_id"])
				require.NotContains(t, metadata, "x-codex-installation-id")
			}
			require.Equal(t, sessionID, originalHeaders.Get("session-id"))
		})
	}
}
