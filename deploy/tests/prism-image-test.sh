#!/usr/bin/env bash
# Disposable Linux integration test. Never uses real credentials or databases.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
work=$(mktemp -d)
project="prism-test-$$"
image="sub2api-prism-test:$$"
cleanup() {
  docker compose -p "$project" -f "$work/deployment/compose.yml" down -v >/dev/null 2>&1 || true
  docker image rm "$image" >/dev/null 2>&1 || true
  rm -rf -- "$work"
}
trap cleanup EXIT
mkdir -p "$work/context/backend" "$work/context/deploy" "$work/deployment/data"
cp "$root/Dockerfile.goreleaser" "$work/context/Dockerfile"
cp -R "$root/prism-adapter" "$work/context/"
cp -R "$root/deploy/prism" "$work/context/deploy/"
cp -R "$root/backend/resources" "$work/context/backend/"
cat > "$work/context/sub2api" <<'PY'
#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')
HTTPServer(('0.0.0.0', 8080), Handler).serve_forever()
PY
chmod +x "$work/context/sub2api"
docker build -t "$image" "$work/context"
printf 'original-state\n' > "$work/deployment/data/sentinel"
cat > "$work/deployment/compose.yml" <<YAML
services:
  sub2api:
    image: $image
    # Simulate a container panel retaining the old entrypoint and command.
    entrypoint: ["/app/docker-entrypoint.sh"]
    command: ["/app/sub2api"]
    security_opt: ["no-new-privileges:true"]
    volumes: ["./data:/app/data"]
    environment:
      DATABASE_PASSWORD: '\${TEST_DB_PASSWORD:?}'
      JWT_SECRET: '\${TEST_JWT_SECRET:?}'
  postgres:
    image: $image
    entrypoint: ["sleep", "infinity"]
    healthcheck: {test: ["CMD", "true"], interval: 1s}
YAML
printf 'TEST_DB_PASSWORD=fixture-only\nTEST_JWT_SECRET=fixture-only-token\n' > "$work/deployment/.env"
cd "$work/deployment"
compose=(docker compose -p "$project" -f compose.yml)
"${compose[@]}" up -d --wait --wait-timeout 90
# This must start before running the migration helper: merely pulling the
# standard image must not strand panels that retained the original entrypoint.
echo 'Legacy explicit entrypoint and command started successfully'
app=$("${compose[@]}" ps -q sub2api)
other=$("${compose[@]}" ps -q postgres)
bash "$root/deploy/upgrade-prism.sh" --container "$app" --local-image "$image"
[[ $("${compose[@]}" ps -q postgres) == "$other" ]]
[[ $(cat data/sentinel) == original-state ]]
# The persisted file still uses .env expressions, not expanded secrets.
grep -Fq '${TEST_DB_PASSWORD:?' compose.yml
if grep -q 'fixture-only' compose.yml; then echo 'Unexpected expanded secret' >&2; exit 1; fi
"${compose[@]}" exec -T sub2api python3 - <<'PY'
import json
from pathlib import Path
import time
import urllib.request
key = Path('/app/data/prism/bridge.key').read_text().strip()
def request(path, method='GET'):
    req = urllib.request.Request('http://127.0.0.1:8320/' + path, method=method, headers={'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.load(response)
assert request('status')['state'] == 'stopped'
request('start', 'POST')
for _ in range(40):
    if request('status')['state'] == 'running': break
    time.sleep(1)
else: raise RuntimeError('Managed adapter did not become ready')
request('stop', 'POST')
assert request('status')['state'] == 'stopped'
print('Managed service controls passed')
PY
"${compose[@]}" restart sub2api
"${compose[@]}" up -d --no-deps --wait --wait-timeout 90 sub2api
# A repeated upgrade must work without moving data or changing another service.
bash "$root/deploy/upgrade-prism.sh" --container "$("${compose[@]}" ps -q sub2api)" --local-image "$image"
[[ $("${compose[@]}" ps -q postgres) == "$other" ]]
[[ $(cat data/sentinel) == original-state ]]
echo 'Bundled browser image and existing-deployment upgrade passed'
