"""Prepare and validate a single existing Compose deployment without host Python."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import secrets
import shutil
import sys

import yaml

ALLOWED = {'image', 'build', 'pull_policy', 'security_opt', 'init', 'shm_size',
           'stop_grace_period', 'entrypoint', 'command'}


def transform(config, image, profile):
    result = copy.deepcopy(config)
    if not isinstance(result, dict) or result.get('include'):
        raise ValueError('Compose include requires an explicit upgrade of its source files')
    services = result.get('services', {})
    app = services.get('sub2api')
    if not isinstance(app, dict) or app.get('extends'):
        raise ValueError('A directly configured sub2api service is required')
    if app.get('privileged') or app.get('read_only') or app.get('network_mode') == 'host':
        raise ValueError('This helper requires a writable, non-privileged container with isolated networking')
    if app.get('entrypoint') not in (None, ['/app/docker-entrypoint.sh'], '/app/docker-entrypoint.sh',
                                    ['/app/managed-entrypoint.sh'], '/app/managed-entrypoint.sh'):
        raise ValueError('Custom entrypoint requires review before upgrading')
    if app.get('command') not in (None, [], ['/app/sub2api'], '/app/sub2api'):
        raise ValueError('Custom command requires review before upgrading')
    if app.get('user') not in (None, '', '0', '0:0', 'root', '1000', '1000:1000', 'pwuser'):
        raise ValueError('Custom container user requires review before upgrading')
    options = app.get('security_opt', [])
    if not isinstance(options, list):
        raise ValueError('Unexpected security_opt configuration')
    kept = []
    for option in options:
        if option in ('no-new-privileges:true', 'no-new-privileges=true', 'no-new-privileges'):
            continue
        if str(option).startswith(('seccomp=', 'seccomp:')):
            if option not in ('seccomp=default', 'seccomp=./prism/seccomp_profile.json',
                              'seccomp=./prism/seccomp-sub2api.json'):
                raise ValueError('Custom seccomp policy requires review before upgrading')
            continue
        kept.append(option)
    app.update(image=image, init=True, shm_size='256m', stop_grace_period='45s',
               security_opt=[*kept, 'seccomp=' + profile])
    for name in ('build', 'pull_policy', 'entrypoint', 'command'):
        app.pop(name, None)
    return result


def safe_file(directory, name):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Expected a file name in the existing deployment directory')
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise ValueError('Compose source must be a regular file, not a symbolic link')
    return path


def prepare(directory, name, image):
    source = safe_file(directory, name)
    config = yaml.safe_load(source.read_text(encoding='utf-8-sig'))
    result = transform(config, image, './prism/seccomp-sub2api.json')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = directory / ('.prism-upgrade-' + stamp + '-' + secrets.token_hex(3))
    backup.mkdir(mode=0o700)
    shutil.copy2(source, backup / 'original.yml')
    (backup / 'original.yml').chmod(0o600)
    candidate = backup / 'candidate.yml'
    candidate.write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True), encoding='utf-8')
    candidate.chmod(0o600)
    (backup / 'metadata.json').write_text(json.dumps({'source': name, 'image': image}), encoding='utf-8')
    print(backup.name)


def validate(before, after, live):
    old, new = copy.deepcopy(before), copy.deepcopy(after)
    for config in (old, new):
        app = config['services']['sub2api']
        for key in ALLOWED:
            app.pop(key, None)
    if old != new:
        raise ValueError('Resolved deployment changed outside the permitted application runtime fields')
    app = before['services']['sub2api']
    environment = app.get('environment', {})
    if (environment.get('GATEWAY_PRISM_BROWSER_API_KEY') or environment.get('GATEWAY_PRISM_BROWSER_BASE_URL')) and not environment.get('GATEWAY_PRISM_BROWSER_MANAGEMENT_URL'):
        raise ValueError('An external Prism service is configured; migrate its saved state explicitly first')
    mounts = app.get('volumes', [])
    if not any(m.get('target') == '/app/data' and not m.get('read_only') for m in mounts):
        raise ValueError('A writable persistent /app/data mount is required')
    live_env = dict(entry.split('=', 1) for entry in live['Config'].get('Env', []) if '=' in entry)
    for key, value in app.get('environment', {}).items():
        if value is not None and str(value) != live_env.get(key):
            raise ValueError('Current Compose environment differs from the running container: ' + key)
    actual = {m['Destination']: m for m in live.get('Mounts', [])}
    for mount in mounts:
        if mount['type'] not in ('bind', 'volume'):
            continue
        expected = mount['source']
        current = actual.get(mount['target'], {})
        if mount['type'] == 'volume':
            expected = before.get('volumes', {}).get(expected, {}).get('name', expected)
            found = current.get('Name')
        else:
            found = current.get('Source')
        if expected != found or current.get('RW') != (not mount.get('read_only', False)):
            raise ValueError('Current Compose data mount differs from the running container')


def install(directory, name, backup_name):
    source = safe_file(directory, name)
    if Path(backup_name).name != backup_name or not backup_name.startswith('.prism-upgrade-'):
        raise ValueError('Invalid backup directory')
    backup = directory / backup_name
    if backup.is_symlink():
        raise ValueError('Backup directory must not be a symbolic link')
    if source.read_bytes() != (backup / 'original.yml').read_bytes():
        raise ValueError('Compose source changed during upgrade; refusing to overwrite it')
    profile_dir = directory / 'prism'
    if profile_dir.is_symlink():
        raise ValueError('Prism profile directory must not be a symbolic link')
    profile_dir.mkdir(exist_ok=True)
    profile = profile_dir / 'seccomp-sub2api.json'
    if profile.is_symlink():
        raise ValueError('Seccomp profile must not be a symbolic link')
    shutil.copyfile(Path(__file__).with_name('seccomp_profile.json'), profile)
    profile.chmod(0o644)
    candidate = backup / 'candidate.yml'
    temp = directory / (backup_name + '.yml')
    with temp.open('xb') as output:
        output.write(candidate.read_bytes())
        output.flush()
        os.fsync(output.fileno())
    temp.chmod(0o600)
    if hasattr(os, 'chown'):
        os.chown(temp, source.stat().st_uid, source.stat().st_gid)
    os.replace(temp, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'validate', 'install'])
    parser.add_argument('--directory', default='/deployment')
    parser.add_argument('--file')
    parser.add_argument('--image')
    parser.add_argument('--backup')
    args = parser.parse_args()
    directory = Path(args.directory)
    if args.action == 'prepare':
        prepare(directory, args.file, args.image)
    elif args.action == 'install':
        install(directory, args.file, args.backup)
    else:
        backup = directory / args.backup
        validate(json.loads((backup / 'before.json').read_text()),
                 json.loads((backup / 'after.json').read_text()),
                 json.loads((backup / 'container.json').read_text())[0])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Never echo parsed configuration or secrets in parser exceptions.
        if isinstance(exc, ValueError):
            print(str(exc), file=sys.stderr)
        else:
            print('Upgrade configuration check failed (' + type(exc).__name__ + ')', file=sys.stderr)
        raise SystemExit(1)
