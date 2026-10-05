"""Offline browser check: no accounts, credentials, or external requests."""
import os
import sys
sys.path.insert(0, '/opt/sub2api/prism-adapter')
from container_runtime import chromium_paths
from playwright.sync_api import sync_playwright


def main():
    if os.geteuid() == 0:
        raise SystemExit('Browser check must run as an ordinary user')
    chrome, helper = chromium_paths()
    os.environ['CHROME_DEVEL_SANDBOX'] = str(helper)
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=str(chrome), headless=True, chromium_sandbox=True)
        try:
            page = browser.new_page()
            page.set_content('<title>Sub2API browser check</title><p>ready</p>')
            if page.title() != 'Sub2API browser check':
                raise RuntimeError('Unexpected browser result')
            page.goto('chrome://sandbox')
            rows = page.locator('tr').all_inner_texts()
            if not any('Seccomp-BPF sandbox' in row and row.split()[-1] == 'Yes' for row in rows):
                raise RuntimeError('Browser sandbox status could not be verified')
        finally:
            browser.close()
    print('Bundled browser and sandbox check passed')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('Bundled browser cannot start with sandboxing on this host; application has not been replaced.', file=sys.stderr)
        raise SystemExit(1)
