import argparse
import ipaddress
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess


def run(*args):
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--public-ip', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--interface', default='eth0')
    parser.add_argument('--port', type=int, default=18443)
    options = parser.parse_args()
    if os.getuid() != 0:
        raise SystemExit('Run as root')
    public = ipaddress.IPv4Address(options.public_ip)
    prefix = ipaddress.IPv6Network(options.prefix)
    if prefix.prefixlen < 64 or prefix.prefixlen > 96 or not prefix.network_address.is_global:
        raise SystemExit('Use a dedicated global /64 through /96 subnet')
    if not 1024 <= options.port <= 65535:
        raise SystemExit('Invalid port')
    os.umask(0o077)
    root = Path('/opt/sub2api-turn-state-egress')
    config = Path('/etc/sub2api-turn-state-egress')
    data = Path('/var/lib/sub2api-turn-state-egress')
    if (config/'config.json').exists():
        raise SystemExit('Existing installation detected; refusing to replace it automatically')
    for command in ('python3', 'openssl', 'ip', 'systemctl'):
        if not shutil.which(command):
            raise SystemExit('Required command missing: ' + command)
    for directory in (root, config, data):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    if source != root:
        shutil.copyfile(source/'pool.py', root/'pool.py')
    shutil.copyfile(source/'sub2api-turn-state-egress.service', '/etc/systemd/system/sub2api-turn-state-egress.service')
    run('openssl', 'req', '-x509', '-newkey', 'rsa:3072', '-nodes', '-days', '1825',
        '-subj', '/CN=Sub2API Turn State Egress CA', '-keyout', str(config/'ca.key'),
        '-out', str(config/'ca.crt'), '-addext', 'basicConstraints=critical,CA:TRUE')
    run('openssl', 'req', '-new', '-newkey', 'rsa:2048', '-nodes', '-subj', '/CN=Sub2API Turn State Egress',
        '-keyout', str(config/'server.key'), '-out', str(config/'server.csr'))
    (config/'server.ext').write_text(f'subjectAltName=IP:{public}\nextendedKeyUsage=serverAuth\nbasicConstraints=CA:FALSE\n')
    run('openssl', 'x509', '-req', '-in', str(config/'server.csr'), '-CA', str(config/'ca.crt'),
        '-CAkey', str(config/'ca.key'), '-CAcreateserial', '-out', str(config/'server.crt'),
        '-days', '825', '-extfile', str(config/'server.ext'))
    value = {'public_url': f'https://{public}:{options.port}', 'interface': options.interface,
             'prefix': str(prefix), 'listen': '0.0.0.0', 'port': options.port,
             'api_token': secrets.token_urlsafe(48), 'tls_cert': str(config/'server.crt'),
             'tls_key': str(config/'server.key'), 'data_dir': str(data), 'ready_target': 24,
             'batch_size': 6, 'lease_seconds': 120, 'max_leases': 32, 'max_connections': 64,
             'allowed_targets': ['chatgpt.com:443', 'api6.ipify.org:443']}
    (config/'config.json').write_text(json.dumps(value, indent=2))
    (config/'sub2api.env').write_text('TURN_STATE_POOL_URL='+value['public_url']+'\nTURN_STATE_POOL_TOKEN='+value['api_token']+'\nTURN_STATE_POOL_CA_FILE=/etc/sub2api-turn-state-egress/ca.crt\n')
    run('systemctl', 'daemon-reload')
    run('systemctl', 'enable', '--now', 'sub2api-turn-state-egress')
    run('systemctl', 'is-active', 'sub2api-turn-state-egress')
    print('Installed independent service. Private connection settings: /etc/sub2api-turn-state-egress/sub2api.env')


if __name__ == '__main__':
    main()
