"""Generate a per-deployment bridge secret without printing or replacing it."""
import argparse
import os
from pathlib import Path
import secrets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    path = args.output.resolve()
    key = secrets.token_urlsafe(48)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise SystemExit('Environment file already exists; it has not been changed') from None
    with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
        stream.write('PRISM_ADAPTER_API_KEY=' + key + '\n')
    print('Private bridge configuration created:', path)


if __name__ == '__main__':
    main()
