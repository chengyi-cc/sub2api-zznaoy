# =============================================================================
# Sub2API Multi-Stage Dockerfile
# =============================================================================
# Stage 1: Build frontend
# Stage 2: Build Go backend with embedded frontend
# Stage 3: Final minimal image
# =============================================================================

ARG NODE_IMAGE=node:24-alpine
ARG GOLANG_IMAGE=golang:1.27.0-alpine
ARG ALPINE_IMAGE=alpine:3.20
ARG GOPROXY=https://goproxy.cn,direct
ARG GOSUMDB=sum.golang.google.cn

# -----------------------------------------------------------------------------
# Stage 1: Frontend Builder
# -----------------------------------------------------------------------------
FROM ${NODE_IMAGE} AS frontend-builder

WORKDIR /app/frontend
ENV NODE_OPTIONS=--max-old-space-size=1536

# Install pnpm. Keep this pinned to the lockfile-compatible major version so
# Docker builds remain reproducible when pnpm changes config validation rules.
RUN corepack enable && corepack prepare pnpm@9.15.9 --activate

# Install dependencies first (better caching)
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

# Copy frontend source and build.
# LegalDocumentView.vue (admin-compliance gate) build-time imports
# ../../../../docs/legal/*.md?raw, so docs/legal/ must sit beside frontend/
# in the image (WORKDIR /app/frontend -> resolves to /app/docs/legal/*.md).
# Copy only that subtree to keep the build dependency minimal.
COPY frontend/ ./
COPY docs/legal/ /app/docs/legal/
RUN pnpm run build

# -----------------------------------------------------------------------------
# Stage 2: Backend Builder
# -----------------------------------------------------------------------------
FROM ${GOLANG_IMAGE} AS backend-builder

# Build arguments for version info (set by CI)
ARG VERSION=
ARG COMMIT=docker
ARG DATE
ARG GOPROXY
ARG GOSUMDB

ENV GOPROXY=${GOPROXY}
ENV GOSUMDB=${GOSUMDB}

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

WORKDIR /app/backend

# Copy go mod files first (better caching)
COPY backend/go.mod backend/go.sum ./
RUN go mod download

# Copy backend source first
COPY backend/ ./

# Copy frontend dist from previous stage (must be after backend copy to avoid being overwritten)
COPY --from=frontend-builder /app/backend/internal/web/dist ./internal/web/dist

# Build the binary (BuildType=release for CI builds, embed frontend)
# Version precedence: build arg VERSION > exact git tag > cmd/server/VERSION
RUN VERSION_VALUE="${VERSION}" && \
    if [ -z "${VERSION_VALUE}" ]; then VERSION_VALUE="$(./scripts/resolve-version.sh)"; fi && \
    DATE_VALUE="${DATE:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}" && \
    CGO_ENABLED=0 GOOS=linux go build \
    -tags embed \
    -ldflags="-s -w -X main.Version=${VERSION_VALUE} -X main.Commit=${COMMIT} -X main.Date=${DATE_VALUE} -X main.BuildType=release" \
    -o /app/sub2api \
    ./cmd/server

# -----------------------------------------------------------------------------
# Stage 3: Bundled browser runtime (same as the root Dockerfile)
# -----------------------------------------------------------------------------
FROM mcr.microsoft.com/playwright/python:v1.63.0-noble AS prism
USER root
ENV HOME=/home/pwuser
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gosu gnupg tzdata && \
    mkdir -p /usr/share/postgresql-common/pgdg && \
    curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc && \
    echo 'deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt noble-pgdg main' > /etc/apt/sources.list.d/pgdg.list && \
    apt-get update && apt-get install -y --no-install-recommends postgresql-client-18 && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /opt/sub2api/prism-adapter
COPY prism-adapter/requirements.txt ./
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt
COPY prism-adapter/ ./
RUN python3 container_runtime.py --prepare-sandbox
COPY deploy/prism/ /opt/sub2api/deploy/prism/
RUN pip install --no-cache-dir --only-binary=:all: -r /opt/sub2api/deploy/prism/requirements-upgrade.txt
LABEL io.sub2api.prism.managed="1"
WORKDIR /app
COPY --from=backend-builder /app/sub2api /app/sub2api
COPY --from=backend-builder /app/backend/resources /app/resources
COPY deploy/prism/managed-entrypoint.sh /app/managed-entrypoint.sh
# Existing container managers can retain the old explicit entrypoint on update.
# Keep both paths functional, including the old /app/sub2api command argument.
RUN mkdir -p /app/data && chown pwuser:pwuser /app/data && chmod +x /app/managed-entrypoint.sh && \
    ln -s /app/managed-entrypoint.sh /app/docker-entrypoint.sh
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fsS --max-time 5 http://127.0.0.1:${SERVER_PORT:-8080}/health > /dev/null || exit 1
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["/app/sub2api"]
