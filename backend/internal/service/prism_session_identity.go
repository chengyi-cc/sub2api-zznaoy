package service

import (
	"errors"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
)

// Adapted from ranxi2001/sub2api session_id.go at 0ae36e5019, scoped to Prism.
// Never change ordinary gateway affinity while adding a new browser channel.
func prismBrowserConversationIdentity(c *gin.Context, body []byte) (string, string, error) {
	kinds := []struct {
		kind, path string
		headers    []string
	}{
		{"thread", "client_metadata.thread_id", []string{"conversation_id", "thread_id", "thread-id", codeBuddyConversationHeader}},
		{"session", "client_metadata.session_id", []string{"session_id", "session-id", openCodeSessionIDHeader, openCodeNativeSessionHeader}},
	}
	for _, entry := range kinds {
		for _, name := range entry.headers {
			if len(c.Request.Header.Values(name)) > 1 {
				return "", "", errors.New("prism conversation identity headers must not be repeated")
			}
		}
	}
	for _, entry := range kinds {
		identity := ""
		accept := func(raw string) error {
			if strings.TrimSpace(raw) == "" {
				return nil
			}
			value := sanitizeSessionID(raw)
			if value == "" || (identity != "" && identity != value) {
				return errors.New("prism conversation identity is invalid or conflicting")
			}
			identity = value
			return nil
		}
		for _, name := range entry.headers {
			if err := accept(c.GetHeader(name)); err != nil {
				return "", "", err
			}
		}
		value := gjson.GetBytes(body, entry.path)
		if value.Exists() {
			if value.Type != gjson.String {
				return "", "", errors.New("prism conversation identity must be a string")
			}
			if err := accept(value.String()); err != nil {
				return "", "", err
			}
		}
		if identity != "" {
			return entry.kind, identity, nil
		}
	}
	return "", "", nil
}
