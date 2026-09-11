package service

import "context"

// HTTPUpstreamProfile marks HTTP upstream requests that need provider-specific
// transport policy.
type HTTPUpstreamProfile string

const (
	HTTPUpstreamProfileDefault    HTTPUpstreamProfile = ""
	HTTPUpstreamProfileOpenAI     HTTPUpstreamProfile = "openai"
	HTTPUpstreamProfileGrok       HTTPUpstreamProfile = "grok"
	HTTPUpstreamProfileLongStream HTTPUpstreamProfile = "long_stream"
)

type httpUpstreamProfileContextKey struct{}
type httpUpstreamDisableRedirectsContextKey struct{}
type httpUpstreamPublicHostsOnlyContextKey struct{}
type codexRequestCompressionContextKey struct{}

// WithHTTPUpstreamProfile injects an upstream transport profile into ctx.
func WithHTTPUpstreamProfile(ctx context.Context, profile HTTPUpstreamProfile) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	if profile == HTTPUpstreamProfileDefault {
		return ctx
	}
	return context.WithValue(ctx, httpUpstreamProfileContextKey{}, profile)
}

// HTTPUpstreamProfileFromContext resolves the upstream transport profile from ctx.
func HTTPUpstreamProfileFromContext(ctx context.Context) HTTPUpstreamProfile {
	if ctx == nil {
		return HTTPUpstreamProfileDefault
	}
	profile, ok := ctx.Value(httpUpstreamProfileContextKey{}).(HTTPUpstreamProfile)
	if !ok {
		return HTTPUpstreamProfileDefault
	}
	switch profile {
	case HTTPUpstreamProfileOpenAI, HTTPUpstreamProfileGrok, HTTPUpstreamProfileLongStream:
		return profile
	default:
		return HTTPUpstreamProfileDefault
	}
}

// WithHTTPUpstreamRedirectsDisabled prevents credential-bearing probes from
// following redirects through the shared upstream client.
func WithHTTPUpstreamRedirectsDisabled(ctx context.Context) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, httpUpstreamDisableRedirectsContextKey{}, true)
}

func HTTPUpstreamRedirectsDisabled(ctx context.Context) bool {
	return ctx != nil && ctx.Value(httpUpstreamDisableRedirectsContextKey{}) == true
}

// WithHTTPUpstreamPublicHostsOnly marks a request whose destination, and every
// redirect hop after it, must resolve to a public address. The shared upstream
// client enforces it regardless of the security.url_allowlist configuration;
// use it for fetches whose URL comes from an untrusted upstream response.
func WithHTTPUpstreamPublicHostsOnly(ctx context.Context) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, httpUpstreamPublicHostsOnlyContextKey{}, true)
}

func HTTPUpstreamPublicHostsOnly(ctx context.Context) bool {
	return ctx != nil && ctx.Value(httpUpstreamPublicHostsOnlyContextKey{}) == true
}

// WithCodexRequestCompression marks a request for configured zstd body encoding
// (zstd 是一种请求体压缩格式). Both core and plugin transports apply it with
// PrepareCodexRequestBody before sending headers or body bytes.
func WithCodexRequestCompression(ctx context.Context) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, codexRequestCompressionContextKey{}, true)
}

func CodexRequestCompressionEnabled(ctx context.Context) bool {
	return ctx != nil && ctx.Value(codexRequestCompressionContextKey{}) == true
}
