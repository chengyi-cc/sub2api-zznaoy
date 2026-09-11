package service

import (
	"container/list"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/openai"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// Synthesized sessions are process-local, bounded, and expire after inactivity.
// Anonymous requests only share an identity within their own gateway context.
const codexMachineSessionLimit = 8192
const codexMachineSessionTTL = 24 * time.Hour
const codexMachineAnonymousSeedKey = "codex_machine_anonymous_seed"

type codexMachineSession struct {
	key             [32]byte
	threadID        string
	contextWindowID string
	lastUsed        time.Time
}

type codexMachineSessionStore struct {
	mutex   sync.Mutex
	entries map[[32]byte]*list.Element
	order   list.List
}

var codexMachineSessions codexMachineSessionStore

func (store *codexMachineSessionStore) resolve(key [32]byte, now time.Time) codexMachineSession {
	store.mutex.Lock()
	defer store.mutex.Unlock()
	if store.entries == nil {
		store.entries = make(map[[32]byte]*list.Element)
	}
	for oldest := store.order.Back(); oldest != nil; oldest = store.order.Back() {
		entry := oldest.Value.(codexMachineSession)
		if now.Sub(entry.lastUsed) < codexMachineSessionTTL {
			break
		}
		delete(store.entries, entry.key)
		store.order.Remove(oldest)
	}
	if element := store.entries[key]; element != nil {
		entry := element.Value.(codexMachineSession)
		entry.lastUsed = now
		element.Value = entry
		store.order.MoveToFront(element)
		return entry
	}
	entry := codexMachineSession{
		key: key, threadID: uuid.Must(uuid.NewV7()).String(),
		contextWindowID: uuid.Must(uuid.NewV7()).String(), lastUsed: now,
	}
	store.entries[key] = store.order.PushFront(entry)
	if store.order.Len() > codexMachineSessionLimit {
		oldest := store.order.Back()
		delete(store.entries, oldest.Value.(codexMachineSession).key)
		store.order.Remove(oldest)
	}
	return entry
}

type codexMachineIdentity struct {
	ids             *codexFingerprintIDs
	threadID        string
	contextWindowID string
	turnID          string
	windowID        string
	turnMetadata    string
}

func codexMachineString(value gjson.Result) string {
	if value.Type != gjson.String {
		return ""
	}
	return strings.TrimSpace(value.String())
}

func codexMachineOrdinaryTurn(headers http.Header, body []byte) bool {
	if len(body) == 0 || !gjson.ValidBytes(body) || !gjson.ParseBytes(body).IsObject() {
		return false
	}
	if event := codexMachineString(gjson.GetBytes(body, "type")); event != "" && event != "response.create" {
		return false
	}
	if generate := gjson.GetBytes(body, "generate"); generate.Type == gjson.False {
		return false
	}
	for _, raw := range []string{headers.Get(openAIWSTurnMetadataHeader), codexMachineString(gjson.GetBytes(body, "client_metadata."+openAIWSTurnMetadataHeader))} {
		if kind := codexMachineString(gjson.Get(raw, "request_kind")); kind != "" && kind != "turn" {
			return false
		}
	}
	return true
}

func codexMachineConversationHint(headers http.Header, body []byte) string {
	// A thread is narrower than a client session: two body threads sharing a
	// session header must remain separate conversations.
	for _, value := range []string{
		strings.TrimSpace(headers.Get("thread-id")),
		codexMachineString(gjson.GetBytes(body, "client_metadata.thread_id")),
		strings.TrimSpace(headers.Get("session-id")),
		strings.TrimSpace(headers.Get("session_id")),
		strings.TrimSpace(headers.Get("x-session-id")),
		codexMachineString(gjson.GetBytes(body, "client_metadata.session_id")),
		codexMachineString(gjson.GetBytes(body, "prompt_cache_key")),
		strings.TrimSpace(headers.Get("x-client-request-id")),
	} {
		if value != "" {
			return value
		}
	}
	return ""
}

func (s *OpenAIGatewayService) resolveCodexFingerprintForRequest(c *gin.Context, account *Account, body []byte) *codexFingerprintIDs {
	var headers http.Header
	if c != nil && c.Request != nil {
		headers = c.Request.Header
	}
	ids := resolveCodexFingerprintIDsFromRequest(account, headers)
	stampCodexMachineSandboxTag(ids, s.codexIdentityOverrideUA(account))
	if ids == nil || ids.mode != codexFingerprintMachine || account == nil || !account.UsesOpenAICodexProtocol() ||
		openai.IsCodexOfficialClientByHeaders(headers.Get("User-Agent"), headers.Get("originator")) ||
		isOpenAIResponsesCompactPath(c) || !codexMachineOrdinaryTurn(headers, body) {
		return ids
	}
	hint := codexMachineConversationHint(headers, body)
	if hint == "" {
		if c != nil {
			hint = c.GetString(codexMachineAnonymousSeedKey)
		}
		if hint == "" {
			hint = uuid.NewString()
			if c != nil {
				c.Set(codexMachineAnonymousSeedKey, hint)
			}
		}
	}
	digest := hmac.New(sha256.New, []byte(ids.machineSeed))
	fmt.Fprintf(digest, "codex-machine-session:v1:account:%d:user:%d:hint:%s", account.ID, getAPIKeyIDFromContext(c), hint)
	var key [32]byte
	copy(key[:], digest.Sum(nil))
	session := codexMachineSessions.resolve(key, time.Now())
	ids.machineIdentity = newCodexMachineIdentity(ids, session.threadID, session.contextWindowID, headers, body)
	return ids
}

func newCodexMachineIdentity(ids *codexFingerprintIDs, threadID, contextWindowID string, headers http.Header, body []byte) *codexMachineIdentity {
	identity := &codexMachineIdentity{
		ids: ids, threadID: threadID, contextWindowID: contextWindowID,
		turnID: uuid.Must(uuid.NewV7()).String(), windowID: threadID + ":0",
	}
	raw := codexMachineString(gjson.GetBytes(body, "client_metadata."+openAIWSTurnMetadataHeader))
	if raw == "" {
		raw = headers.Get(openAIWSTurnMetadataHeader)
	}
	metadata := make(map[string]any)
	decoder := json.NewDecoder(strings.NewReader(rewriteCodexMachineTurnMetadata(raw, ids)))
	decoder.UseNumber()
	if decoder.Decode(&metadata) != nil || metadata == nil {
		metadata = make(map[string]any)
	}
	for key, value := range map[string]any{
		"installation_id": ids.installationID, "session_id": threadID, "thread_id": threadID,
		"turn_id": identity.turnID, "window_id": identity.windowID, "context_window_id": contextWindowID,
		"agent_name": "/root", "thread_source": "user", "request_kind": "turn", "turn_started_at_unix_ms": time.Now().UnixMilli(),
	} {
		metadata[key] = value
	}
	encoded, _ := json.Marshal(metadata)
	identity.turnMetadata = string(encoded)
	return identity
}

func (identity *codexMachineIdentity) applyHeaders(headers http.Header) {
	headers.Set("session-id", identity.threadID)
	headers.Set("thread-id", identity.threadID)
	headers.Set("x-client-request-id", identity.threadID)
	headers.Set("x-codex-window-id", identity.windowID)
	headers.Set(openAIWSTurnMetadataHeader, identity.turnMetadata)
	headers.Del("session_id")
	headers.Del("conversation_id")
	headers.Del("x-codex-installation-id")
	if raw := headers.Get("x-codex-parent-thread-id"); raw != "" {
		headers.Set("x-codex-parent-thread-id", identity.ids.machinePseudonym(raw))
	}
}

func (identity *codexMachineIdentity) applyMetadata(metadata map[string]any) bool {
	changed := false
	// Rewrite optional parent linkage without remapping the already synthesized
	// thread/window values; repeated projection must report no further change.
	if original, ok := metadata["x-codex-parent-thread-id"].(string); ok && strings.TrimSpace(original) != "" {
		next := identity.ids.machinePseudonym(original)
		if original != next {
			metadata["x-codex-parent-thread-id"] = next
			changed = true
		}
	}
	for key, value := range map[string]string{
		"x-codex-installation-id": identity.ids.installationID,
		"session_id":              identity.threadID, "thread_id": identity.threadID, "turn_id": identity.turnID,
		"x-codex-window-id": identity.windowID, openAIWSTurnMetadataHeader: identity.turnMetadata,
	} {
		if original, ok := metadata[key].(string); !ok || original != value {
			metadata[key] = value
			changed = true
		}
	}
	return changed
}

func (identity *codexMachineIdentity) applyBody(body map[string]any) bool {
	metadata, _ := body["client_metadata"].(map[string]any)
	if metadata == nil {
		metadata = make(map[string]any)
	}
	changed := identity.applyMetadata(metadata)
	if changed {
		body["client_metadata"] = metadata
	}
	if cacheKey, _ := body["prompt_cache_key"].(string); cacheKey != identity.threadID {
		body["prompt_cache_key"] = identity.threadID
		changed = true
	}
	return changed
}

func (identity *codexMachineIdentity) applyRawBody(body []byte) ([]byte, bool, error) {
	if !codexMachineOrdinaryTurn(nil, body) {
		return body, false, nil
	}
	metadata := make(map[string]any)
	if existing := gjson.GetBytes(body, "client_metadata"); existing.IsObject() {
		decoder := json.NewDecoder(strings.NewReader(existing.Raw))
		decoder.UseNumber()
		if err := decoder.Decode(&metadata); err != nil {
			return body, false, err
		}
	}
	next := body
	changed := identity.applyMetadata(metadata)
	if changed {
		encoded, err := json.Marshal(metadata)
		if err != nil {
			return body, false, err
		}
		next, err = sjson.SetRawBytes(next, "client_metadata", encoded)
		if err != nil {
			return body, false, err
		}
	}
	if codexMachineString(gjson.GetBytes(body, "prompt_cache_key")) != identity.threadID {
		var err error
		next, err = sjson.SetBytes(next, "prompt_cache_key", identity.threadID)
		if err != nil {
			return body, false, err
		}
		changed = true
	}
	return next, changed, nil
}

func (s *OpenAIGatewayService) advanceCodexMachineWebSocketTurn(c *gin.Context, account *Account, body []byte) {
	ids := stagedCodexFingerprintIDs(c, account)
	if ids == nil || !codexMachineOrdinaryTurn(nil, body) {
		return
	}
	if ids.machineIdentity == nil {
		// A generate:false warmup may precede the first ordinary response.create.
		stageCodexFingerprintIDs(c, s.resolveCodexFingerprintForRequest(c, account, body))
		return
	}
	previous := ids.machineIdentity
	ids.machineIdentity = newCodexMachineIdentity(ids, previous.threadID, previous.contextWindowID, nil, body)
}
