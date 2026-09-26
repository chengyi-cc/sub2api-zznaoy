package service

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	coderws "github.com/coder/websocket"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// Emit each completion only after its request arrives, like a real upstream.
type timezoneTurnConn struct {
	*openAIWSCaptureConn
	completed chan []byte
}

func (c *timezoneTurnConn) WriteJSON(ctx context.Context, value any) error {
	if err := c.openAIWSCaptureConn.WriteJSON(ctx, value); err != nil {
		return err
	}
	c.mu.Lock()
	n := len(c.writes)
	c.mu.Unlock()
	event := []byte(fmt.Sprintf(`{"type":"response.completed","response":{"id":"resp_tz_%d","model":"gpt-5.6-sol","output":[],"usage":{"input_tokens":1,"output_tokens":1}}}`, n))
	select {
	case c.completed <- event:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (c *timezoneTurnConn) WriteFrame(ctx context.Context, _ coderws.MessageType, payload []byte) error {
	return c.WriteJSON(ctx, json.RawMessage(payload))
}

func (c *timezoneTurnConn) ReadMessage(ctx context.Context) ([]byte, error) {
	select {
	case event := <-c.completed:
		return event, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (c *timezoneTurnConn) ReadFrame(ctx context.Context) (coderws.MessageType, []byte, error) {
	event, err := c.ReadMessage(ctx)
	return coderws.MessageText, event, err
}

func TestCodexTimezoneWebSocketEveryTurn(t *testing.T) {
	for _, mode := range []string{OpenAIWSIngressModeCtxPool, OpenAIWSIngressModePassthrough} {
		t.Run(mode, func(t *testing.T) {
			cfg := &config.Config{}
			cfg.Gateway.OpenAIWS.Enabled = true
			cfg.Gateway.OpenAIWS.OAuthEnabled = true
			cfg.Gateway.OpenAIWS.ResponsesWebsocketsV2 = true
			cfg.Gateway.OpenAIWS.ModeRouterV2Enabled = true
			cfg.Gateway.OpenAIWS.MaxConnsPerAccount = 1
			cfg.Gateway.OpenAIWS.MaxIdlePerAccount = 1
			cfg.Gateway.OpenAIWS.ReadTimeoutSeconds = 3
			cfg.Gateway.OpenAIWS.WriteTimeoutSeconds = 3
			capture := &timezoneTurnConn{openAIWSCaptureConn: &openAIWSCaptureConn{}, completed: make(chan []byte, 2)}
			dialer := &openAIWSSingleConnDialer{conn: capture}
			pool := newOpenAIWSConnPool(cfg)
			defer pool.Close()
			pool.setClientDialerForTest(dialer)
			svc := &OpenAIGatewayService{cfg: cfg, cache: &stubGatewayCache{}, toolCorrector: NewCodexToolCorrector(), openaiWSResolver: NewOpenAIWSProtocolResolver(cfg), openaiWSPool: pool, openaiWSPassthroughDialer: dialer}
			lookups := 0
			svc.codexTimezone.lookup = func(context.Context, string) (string, error) { lookups++; return "America/Los_Angeles", nil }
			svc.codexTimezone.now = func() time.Time { return time.Date(2026, 9, 26, 1, 0, 0, 0, time.UTC) }
			account := excelAccount()
			account.Extra = map[string]any{"openai_codex_timezone_rewrite": true, "openai_oauth_responses_websockets_v2_mode": mode}
			done := make(chan error, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				conn, err := coderws.Accept(w, r, nil)
				if err != nil {
					done <- err
					return
				}
				defer conn.CloseNow()
				_, first, err := conn.Read(r.Context())
				if err != nil {
					done <- err
					return
				}
				c, _ := gin.CreateTestContext(httptest.NewRecorder())
				c.Request = r
				done <- svc.ProxyResponsesWebSocketFromClient(r.Context(), c, conn, account, "test-token", first, nil)
			}))
			defer server.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			client, _, err := coderws.Dial(ctx, "ws"+strings.TrimPrefix(server.URL, "http"), nil)
			require.NoError(t, err)
			defer client.CloseNow()
			for i := 0; i < 2; i++ {
				body, err := sjson.SetBytes(timezoneTestBody(t, timezoneTestEnvironment), "type", "response.create")
				require.NoError(t, err)
				require.NoError(t, client.Write(ctx, coderws.MessageText, body))
				_, response, err := client.Read(ctx)
				require.NoError(t, err)
				require.Equal(t, "response.completed", gjson.GetBytes(response, "type").String())
			}
			_ = client.Close(coderws.StatusNormalClosure, "done")
			select {
			case err := <-done:
				if err != nil {
					require.Contains(t, err.Error(), "StatusNormalClosure")
				}
			case <-ctx.Done():
				t.Fatal("websocket did not complete")
			}
			require.Equal(t, 1, lookups, "two turns share one egress lookup")
			capture.mu.Lock()
			writes := append([]map[string]any(nil), capture.writes...)
			capture.mu.Unlock()
			require.Len(t, writes, 2)
			for _, write := range writes {
				encoded, err := json.Marshal(write)
				require.NoError(t, err)
				text := gjson.GetBytes(encoded, `input.#(role=="user").content.0.text`).String()
				require.Contains(t, text, "<timezone>America/Los_Angeles</timezone>")
				require.Contains(t, text, "<current_date>2026-09-25</current_date>")
			}
		})
	}
}
