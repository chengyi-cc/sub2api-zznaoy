#!/bin/sh
set -eu
if [ "$(id -u)" = 0 ]; then
    mkdir -p /app/data/prism
    # Preserve deployments with an additional read-only config.yaml mount.
    chown -R pwuser:pwuser /app/data 2>/dev/null || true
    exec gosu pwuser "$0" "$@"
fi
if [ "${1:-}" = /app/sub2api ]; then
    shift
fi
# Browser dependencies and key setup are owned by the bundled runtime.
# Only the gateway's normal command-line flags are accepted here.
exec python3 /opt/sub2api/prism-adapter/managed_runtime.py "$@"
