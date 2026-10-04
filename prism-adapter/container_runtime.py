"""Locate the matching packaged Chromium without disabling its sandbox."""
import os
from pathlib import Path
import sys
from playwright.sync_api import sync_playwright


def chromium_paths():
    with sync_playwright() as runtime:
        chrome = Path(runtime.chromium.executable_path)
    if not chrome.is_file():
        raise SystemExit('Matching Chromium is missing; install the pinned browser runtime')
    for name in ('chrome-sandbox', 'chrome_sandbox'):
        helper = chrome.parent / name
        if helper.is_file():
            return chrome, helper
    raise SystemExit('Chromium sandbox helper is missing')


def main():
    if not sys.platform.startswith('linux'):
        raise SystemExit('The Prism browser runtime requires Linux')
    chrome, helper = chromium_paths()
    if sys.argv[1:] == ['--prepare-sandbox']:
        if os.geteuid() != 0:
            raise SystemExit('Preparing the sandbox helper requires root at image build time')
        os.chown(helper, 0, 0)
        helper.chmod(0o4755)
        return
    if os.geteuid() == 0:
        raise SystemExit('Run the adapter and its checks as a non-root user')
    os.environ.setdefault('PRISM_ADAPTER_CHROME', str(chrome))
    os.environ.setdefault('CHROME_DEVEL_SANDBOX', str(helper))
    args = sys.argv[1:]
    if args:
        os.execvp(args[0], args)
    os.execv(sys.executable, [sys.executable, str(Path(__file__).with_name('server.py'))])


if __name__ == '__main__':
    main()
