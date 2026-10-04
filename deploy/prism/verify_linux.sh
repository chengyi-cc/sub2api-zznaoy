#!/usr/bin/env bash
set -euo pipefail
PRISM_VERIFY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PRISM_VERIFY_PYTHON="${PRISM_VERIFY_PYTHON:-python3}"
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Prism runtime verification requires Linux.' >&2
  exit 1
fi
cd "$PRISM_VERIFY_DIR"
"$PRISM_VERIFY_PYTHON" -m unittest discover -s prism-adapter -p 'test_*.py' -v
# These browser checks use local fixtures; they do not log into real accounts.
PRISM_VERIFY_CHROME="${PRISM_ADAPTER_CHROME:-$("$PRISM_VERIFY_PYTHON" -c 'from playwright.sync_api import sync_playwright; p=sync_playwright().start(); print(p.chromium.executable_path); p.stop()')}"
"$PRISM_VERIFY_PYTHON" prism-adapter/smoke_browser.py --chrome "$PRISM_VERIFY_CHROME"
"$PRISM_VERIFY_PYTHON" prism-adapter/smoke_client_tools.py --chrome "$PRISM_VERIFY_CHROME"
"$PRISM_VERIFY_PYTHON" prism-adapter/smoke_multiplex.py --chrome "$PRISM_VERIFY_CHROME" --concurrency 5 --model gpt-6.1-sol --effort xhigh --rounds 3
