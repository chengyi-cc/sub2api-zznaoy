# Source and local integration

Upstream: https://github.com/ranxi2001/sub2api

Pinned commit: `0ae36e501952000c5c910a2e616c6e0861f66a49` (production, checked 2026-10-04).
Latest published release at the time: v2.9.8. This source additionally includes the merged Prism fixes in PR #292.

The upstream Python adapter, tests, smoke scripts, requirements, and systemd templates are preserved. Local additions are `Dockerfile`, `.dockerignore`, `container_runtime.py`, and this provenance record. The Go gateway integration is adapted to this repository's account, billing, scheduling, and session interfaces.

The bundled `deploy/prism/seccomp_profile.json` comes from Playwright v1.63.0:
https://github.com/microsoft/playwright/blob/v1.63.0/utils/docker/seccomp_profile.json

Local adapter runtime target: Linux. Do not remove directory synchronization, process resource limits, pending-turn records, tool validation, or the Chromium sandbox to make a different platform appear supported.
