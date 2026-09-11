package service

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

func applyCodexMachineCompactProbeIdentity(payload map[string]any, probeSessionID, sandboxTag string, startedAt time.Time) string {
	turnID := uuid.Must(uuid.NewV7()).String()
	installationID := deriveStableUUIDv4("sub2api:codex-compact-probe:v1:installation:" + probeSessionID)
	windowID := probeSessionID + ":0"
	metadata := map[string]any{
		"installation_id": installationID, "session_id": probeSessionID, "thread_id": probeSessionID,
		"agent_name": "/root", "turn_id": turnID, "window_id": windowID,
		"request_kind": "compaction", "thread_source": "user", "sandbox_mode": "workspace-write",
		"auto_review_enabled": false, "node_repl_auto_review_required": false, "node_repl_disabled": false,
		"turn_started_at_unix_ms": startedAt.UnixMilli(),
		"compaction":              map[string]any{"trigger": "manual", "reason": "user_requested", "implementation": "responses_compaction_v2", "phase": "standalone_turn", "strategy": "memento"},
	}
	if sandboxTag != "" {
		metadata["sandbox"] = sandboxTag
	}
	encoded, _ := json.Marshal(metadata)
	payload["prompt_cache_key"] = probeSessionID
	payload["client_metadata"] = map[string]any{
		"x-codex-installation-id": installationID, "session_id": probeSessionID, "thread_id": probeSessionID,
		"turn_id": turnID, "x-codex-window-id": windowID, "x-codex-turn-metadata": string(encoded),
	}
	return string(encoded)
}

func isCodexMachineIdentityHeader(name string) bool {
	switch name {
	case "session-id", "thread-id", "x-client-request-id", "x-codex-parent-thread-id":
		return true
	default:
		return false
	}
}

func usesCodexMachineFingerprint(account *Account) bool {
	if account == nil || account.GetCodexFingerprintMode() != codexFingerprintMachine {
		return false
	}
	_, valid := codexFingerprintSeed(account.Extra)
	return valid
}

func (ids *codexFingerprintIDs) machinePseudonym(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	ids.machineMutex.Lock()
	defer ids.machineMutex.Unlock()
	if pseudonym, exists := ids.machinePseudonyms[raw]; exists {
		return pseudonym
	}
	mac := hmac.New(sha256.New, []byte(ids.machineSeed))
	mac.Write([]byte("sub2api:codex-machine:v1:" + raw))
	digest := mac.Sum(nil)
	var identifier uuid.UUID
	if original, err := uuid.Parse(raw); err == nil && original.Version() == 7 {
		copy(identifier[:6], original[:6])
		copy(identifier[6:], digest[:10])
		identifier[6] = (identifier[6] & 0x0f) | 0x70
	} else {
		copy(identifier[:], digest[:16])
		identifier[6] = (identifier[6] & 0x0f) | 0x40
	}
	identifier[8] = (identifier[8] & 0x3f) | 0x80
	pseudonym := identifier.String()
	if ids.machinePseudonyms == nil {
		ids.machinePseudonyms = make(map[string]string)
	}
	ids.machinePseudonyms[raw] = pseudonym
	ids.machinePseudonyms[pseudonym] = pseudonym
	return pseudonym
}

func (ids *codexFingerprintIDs) machineWindowPseudonym(raw string) string {
	trimmed := strings.TrimSpace(raw)
	base, suffix := trimmed, ""
	if separator := strings.LastIndexByte(trimmed, ':'); separator >= 0 {
		base, suffix = trimmed[:separator], trimmed[separator:]
	}
	if _, err := uuid.Parse(base); err != nil {
		return raw
	}
	return ids.machinePseudonym(base) + suffix
}

func stampCodexMachineSandboxTag(ids *codexFingerprintIDs, candidateUA string) {
	if ids == nil || ids.mode != codexFingerprintMachine {
		return
	}
	userAgent := resolveCodexOutboundIdentity(candidateUA).userAgent
	start := strings.IndexByte(userAgent, '(')
	if start < 0 {
		return
	}
	rest := userAgent[start+1:]
	end := strings.IndexByte(rest, ')')
	if end < 0 {
		return
	}
	osSegment := strings.TrimSpace(strings.SplitN(rest[:end], ";", 2)[0])
	switch {
	case osSegment == "":
	case strings.HasPrefix(osSegment, "Mac OS"):
		ids.machineSandboxTag = "seatbelt"
	case strings.HasPrefix(osSegment, "Windows"):
		ids.machineSandboxTag = "windows_sandbox"
	default:
		ids.machineSandboxTag = "seccomp"
	}
}

func rewriteCodexMachineSandbox(raw, target string) string {
	if target == "" {
		return raw
	}
	switch raw {
	case "seccomp", "seatbelt", "windows_sandbox", "windows_elevated":
		if target == "windows_sandbox" && (raw == "windows_sandbox" || raw == "windows_elevated") {
			return raw
		}
		return target
	default:
		return raw
	}
}

func dedupeCodexMachineTurnMetadata(raw string) string {
	type field struct{ key, value gjson.Result }
	fields := []field{}
	lastIndex := map[string]int{}
	gjson.Parse(raw).ForEach(func(key, value gjson.Result) bool {
		lastIndex[key.String()] = len(fields)
		fields = append(fields, field{key: key, value: value})
		return true
	})
	if len(lastIndex) == len(fields) {
		return raw
	}
	var result strings.Builder
	result.WriteByte('{')
	written := false
	for index, field := range fields {
		if lastIndex[field.key.String()] != index {
			continue
		}
		if written {
			result.WriteByte(',')
		}
		result.WriteString(field.key.Raw)
		result.WriteByte(':')
		result.WriteString(field.value.Raw)
		written = true
	}
	result.WriteByte('}')
	return result.String()
}

func rewriteCodexMachineTurnMetadata(raw string, ids *codexFingerprintIDs) string {
	if !gjson.Valid(raw) || !gjson.Parse(raw).IsObject() {
		return raw
	}
	rewritten := dedupeCodexMachineTurnMetadata(raw)
	metadata := gjson.Parse(rewritten)
	for _, field := range []string{"installation_id", "session_id", "thread_id", "parent_thread_id", "forked_from_thread_id", "context_window_id", "window_id", "sandbox"} {
		value := metadata.Get(field)
		if value.Type != gjson.String || strings.TrimSpace(value.String()) == "" {
			continue
		}
		original := value.String()
		next := original
		switch field {
		case "installation_id":
			next = ids.installationID
		case "window_id":
			next = ids.machineWindowPseudonym(original)
		case "sandbox":
			next = rewriteCodexMachineSandbox(original, ids.machineSandboxTag)
		default:
			next = ids.machinePseudonym(original)
		}
		if next != original {
			updated, err := sjson.Set(rewritten, field, next)
			if err != nil {
				return raw
			}
			rewritten = updated
		}
	}
	return rewritten
}

func applyCodexMachineClientMetadata(metadata map[string]any, ids *codexFingerprintIDs) bool {
	modified := false
	for _, field := range []string{"x-codex-installation-id", "session_id", "thread_id", "x-codex-parent-thread-id", "x-codex-window-id", "x-codex-turn-metadata"} {
		original, ok := metadata[field].(string)
		if !ok || strings.TrimSpace(original) == "" {
			continue
		}
		var next string
		switch field {
		case "x-codex-installation-id":
			next = ids.installationID
		case "x-codex-window-id":
			next = ids.machineWindowPseudonym(original)
		case "x-codex-turn-metadata":
			next = rewriteCodexMachineTurnMetadata(original, ids)
		default:
			next = ids.machinePseudonym(original)
		}
		if next != original {
			metadata[field] = next
			modified = true
		}
	}
	return modified
}

func applyCodexMachineHeaders(headers http.Header, ids *codexFingerprintIDs) {
	if ids.machineIdentity != nil {
		ids.machineIdentity.applyHeaders(headers)
		return
	}
	if strings.TrimSpace(headers.Get("x-codex-installation-id")) != "" {
		headers.Set("x-codex-installation-id", ids.installationID)
	}
	for _, field := range []string{"session-id", "thread-id", "x-client-request-id", "x-codex-parent-thread-id"} {
		if raw := headers.Get(field); strings.TrimSpace(raw) != "" {
			headers.Set(field, ids.machinePseudonym(raw))
		}
	}
	if raw := headers.Get("x-codex-window-id"); strings.TrimSpace(raw) != "" {
		headers.Set("x-codex-window-id", ids.machineWindowPseudonym(raw))
	}
	headers.Del("conversation_id")
	if strings.TrimSpace(headers.Get("session-id")) != "" || strings.TrimSpace(headers.Get("thread-id")) != "" {
		headers.Del("session_id")
	}
	if raw := headers.Get("x-codex-turn-metadata"); strings.TrimSpace(raw) != "" {
		headers.Set("x-codex-turn-metadata", rewriteCodexMachineTurnMetadata(raw, ids))
	}
}

func (s *OpenAIGatewayService) stageCodexMachineFingerprintIDs(c *gin.Context, account *Account, bodies ...[]byte) {
	stageCodexFingerprintIDs(c, nil)
	if !usesCodexMachineFingerprint(account) {
		return
	}
	var body []byte
	if len(bodies) > 0 {
		body = bodies[0]
	}
	ids := s.resolveCodexFingerprintForRequest(c, account, body)
	stageCodexFingerprintIDs(c, ids)
}

func (s *OpenAIGatewayService) stageCodexMachineFingerprintIDsForCompatBridge(c *gin.Context, account *Account, body []byte) ([]byte, error) {
	s.stageCodexMachineFingerprintIDs(c, account, body)
	next, _, err := applyCodexFingerprintClientMetadataRaw(body, stagedCodexFingerprintIDs(c, account))
	return next, err
}
