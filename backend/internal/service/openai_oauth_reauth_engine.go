package service

import (
	"context"
	infraerrors "github.com/Wei-Shaw/sub2api/internal/pkg/errors"
)

// This port runs account recovery locally. No third-party login endpoint is
// enabled implicitly by an upstream default or by restored configuration.
func (s *OpenAIOAuthReauthService) sessionStudioConfig(context.Context) (string, map[string]string, error) {
	return "", nil, infraerrors.BadRequest("OPENAI_REAUTH_ENGINE_UNAVAILABLE", "This installation supports local re-login only")
}
