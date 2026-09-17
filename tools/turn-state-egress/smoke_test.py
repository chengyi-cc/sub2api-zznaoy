import argparse
import ipaddress
import json
from pathlib import Path
import ssl
import subprocess
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/etc/sub2api-turn-state-egress/config.json')
    options = parser.parse_args()
    config_path = Path(options.config)
    config = json.loads(config_path.read_text())
    certificate = str(config_path.parent/'ca.crt')
    tls = ssl.create_default_context(cafile=certificate)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=tls))

    def call(method, path):
        request = urllib.request.Request(config['public_url']+path, method=method,
                                         headers={'Authorization': 'Bearer '+config['api_token']})
        with opener.open(request, timeout=15) as response:
            return json.load(response)

    leases = []
    try:
        initial = call('GET', '/v1/status')
        print('Initial pool status:', json.dumps(initial))
        for attempt in range(3):
            lease = call('POST', '/v1/leases')
            leases.append(lease)
            curl_config = 'proxy = "'+lease['proxy_url']+'"\nproxy-cacert = "'+certificate+'"\nurl = "https://api6.ipify.org"\n'
            result = subprocess.run(['curl', '--silent', '--show-error', '--fail', '--write-out', '\n%{http_connect} %{response_code}', '--max-time', '25', '--noproxy', '', '--config', '-'],
                                    input=curl_config, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                safe_error = result.stderr.replace(lease['proxy_url'], '[proxy]')[:200]
                raise RuntimeError('Authenticated TLS proxy exit verification failed, curl code '+str(result.returncode)+': '+safe_error+' status='+result.stdout[-30:])
            actual = str(ipaddress.IPv6Address(result.stdout.splitlines()[0].strip()))
            if actual != lease['ipv6']:
                raise RuntimeError('Actual exit differs from leased IPv6')
            print('Verified real exit:', actual)
        if len({lease['ipv6'] for lease in leases}) != len(leases):
            raise RuntimeError('Duplicate exit allocation')
        print('PASS: three distinct leased exits match the externally observed IPv6 addresses')
    finally:
        for lease in leases:
            call('DELETE', '/v1/leases/'+lease['id'])
        print('Final pool status:', json.dumps(call('GET', '/v1/status')))


if __name__ == '__main__':
    main()
