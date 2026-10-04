"""Local, authenticated control plane for the bundled Prism browser service.

Only fixed service operations are exposed. No shell, Docker socket, credentials,
prompts, or raw subprocess output are exposed to the admin UI.
"""
import collections
import datetime
import hmac
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request

STARTUP_EVENTS = {
    'Matching Chromium is missing; install the pinned browser runtime': 'browser_missing',
    'Chromium sandbox helper is missing': 'sandbox_missing',
    'Run the adapter and its checks as a non-root user': 'non_root_required',
    'Prism adapter must run as a non-root user': 'non_root_required',
    'adapter key, Chromium binary, and Chromium sandbox are required': 'adapter_configuration_invalid',
    'Install prism-adapter/requirements.txt before enabling client tools': 'dependencies_missing',
}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    with open(temp, 'w', encoding='utf-8') as handle:
        os.chmod(temp, 0o600)
        json.dump(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def bridge_key(directory):
    path = directory / 'bridge.key'
    try:
        with open(path, 'x', encoding='utf-8') as handle:
            os.chmod(path, 0o600)
            handle.write(secrets.token_urlsafe(48))
    except FileExistsError:
        pass
    key = path.read_text(encoding='utf-8').strip()
    if len(key) < 32 or not key.isascii() or any(c.isspace() for c in key):
        raise ValueError('invalid stored bridge key')
    return key


class Manager:
    def __init__(self, directory, command, env, adapter_port=8319):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.config = self.directory / 'service.json'
        self.command, self.env, self.adapter_port = command, env, adapter_port
        self.lock = threading.RLock()
        self.logs = collections.deque(maxlen=200)
        self.sequence = 0
        self.process = None
        self.state = 'stopped'
        self.desired = False
        self.started = 0
        self.failures = collections.deque()
        self.last_retry = 0
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        if self.config.exists():
            saved = json.loads(self.config.read_text(encoding='utf-8'))
            if type(saved.get('enabled')) is not bool:
                raise ValueError('invalid stored service setting')
            self.desired = saved['enabled']
        self.event('manager_ready')

    def event(self, code):
        # Codes only; never publish browser output, exception messages or tokens.
        with self.lock:
            self.sequence += 1
            self.logs.append({'id': self.sequence, 'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'code': code})

    def capture(self, process):
        try:
            self.capture_lines(process)
        finally:
            process.stdout.close()

    def capture_lines(self, process):
        while True:
            line = process.stdout.readline(4097)
            if not line:
                return
            if len(line) > 4096:
                while line and not line.endswith(b'\n'):
                    line = process.stdout.readline(4097)
                continue
            try:
                data = json.loads(line)
                event = data.get('event') if isinstance(data, dict) else None
                if event in {'prism_adapter_error', 'prism_worker_close_error'}:
                    self.event(event)
            except (ValueError, UnicodeError):
                code = STARTUP_EVENTS.get(line.decode('utf-8', errors='replace').strip())
                if code:
                    self.event(code)

    def healthy(self):
        try:
            with self.opener.open(f'http://127.0.0.1:{self.adapter_port}/health', timeout=1) as reply:
                return reply.status == 200 and json.loads(reply.read(1024)).get('status') == 'ok'
        except Exception:
            return False

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        # Refuse to claim or stop an unrelated adapter already using this port.
        if self.healthy():
            self.state = 'error'
            self.event('port_in_use')
            return
        try:
            self.process = subprocess.Popen(self.command, env=self.env, stdin=subprocess.DEVNULL,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            start_new_session=True)
            self.started = time.monotonic()
            self.state = 'starting'
            self.event('service_starting')
            threading.Thread(target=self.capture, args=(self.process,), daemon=True).start()
        except OSError:
            self.process = None
            self.state = 'error'
            self.event('service_start_failed')

    def stop(self):
        process = self.process
        if process is not None:
            # Give the adapter wrapper time to close Playwright first, then clean
            # up the owned process group if the browser worker is stuck.
            try:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=9)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=2)
            except ProcessLookupError:
                pass
            if os.name == 'posix':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process = None
        self.state = 'stopped'
        self.event('service_stopped')

    def tick(self):
        with self.lock:
            now = time.monotonic()
            if self.process is not None and self.process.poll() is not None:
                self.stop()
                self.state = 'error'
                self.failures.append(now)
                self.event('service_exited')
            while self.failures and now - self.failures[0] > 60:
                self.failures.popleft()
            if self.desired and self.process is None and self.state != 'error':
                self.start()
            elif self.desired and self.process is None and len(self.failures) < 3 and now - self.last_retry >= 10:
                self.last_retry = now
                self.start()
            if self.process is not None and self.process.poll() is None:
                ok = self.healthy()
                new_state = 'running' if ok else ('starting' if now - self.started < 30 else 'error')
                if new_state == 'running' and self.state != 'running':
                    self.event('service_ready')
                if new_state == 'error' and self.state != 'error':
                    self.event('health_failed')
                self.state = new_state

    def snapshot(self, include_logs=False):
        with self.lock:
            result = {'managed': True, 'state': self.state, 'healthy': self.state == 'running', 'desired_enabled': self.desired}
            if include_logs:
                result['logs'] = list(self.logs)
            return result

    def action(self, action):
        with self.lock:
            if action == 'check':
                self.tick()
                self.event('check_ok' if self.state == 'running' else 'check_failed')
                return self.snapshot()
            if action not in {'start', 'stop', 'restart'}:
                raise ValueError('unsupported action')
            enabled = action != 'stop'
            # Persist the intent before affecting the running process.
            atomic_json(self.config, {'enabled': enabled})
            self.desired = enabled
            self.failures.clear()
            if action in {'stop', 'restart'}:
                self.stop()
            if enabled:
                self.start()
            return self.snapshot()


def handler_for(manager, key):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def reply(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            values = self.headers.get_all('Authorization', [])
            if self.headers.get('Origin') or len(values) != 1 or not hmac.compare_digest(values[0].encode(), ('Bearer ' + key).encode()):
                self.reply(401, {'error': 'unauthorized'})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path not in {'/status', '/logs'}:
                self.reply(404, {'error': 'not_found'})
                return
            self.reply(200, manager.snapshot(self.path == '/logs'))

        def do_POST(self):
            if not self.authorized():
                return
            if self.path not in {'/start', '/stop', '/restart', '/check'}:
                self.reply(404, {'error': 'not_found'})
                return
            # No request bodies or arbitrary command/path parameters accepted.
            if self.headers.get('Transfer-Encoding') or self.headers.get('Content-Length', '0') != '0':
                self.reply(400, {'error': 'body_not_allowed'})
                return
            try:
                result = manager.action(self.path[1:])
            except Exception:
                manager.event('control_failed')
                self.reply(503, {'error': 'control_failed'})
                return
            self.reply(200, result)
    return Handler


def main():
    if not sys.platform.startswith('linux') or os.geteuid() == 0:
        raise SystemExit('The bundled Prism runtime requires a non-root Linux user')
    directory = Path(os.environ.get('PRISM_MANAGED_STATE_DIR', '/app/data/prism'))
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = bridge_key(directory)
    env = os.environ.copy()
    # Always locate the matching bundled browser, ignoring old host paths.
    env.pop('PRISM_ADAPTER_CHROME', None)
    env.pop('CHROME_DEVEL_SANDBOX', None)
    env.update({'PRISM_ADAPTER_API_KEY': key, 'PRISM_ADAPTER_PORT': '8319',
                'PRISM_ADAPTER_STATE_DIR': str(directory / 'adapter'),
                'PRISM_ADAPTER_MODE': 'browser', 'PRISM_ADAPTER_CLIENT_TOOLS_ENABLED': 'true',
                'GATEWAY_PRISM_BROWSER_ENABLED': 'true',
                'GATEWAY_PRISM_BROWSER_BASE_URL': 'http://127.0.0.1:8319/v1',
                'GATEWAY_PRISM_BROWSER_API_KEY': key,
                'GATEWAY_PRISM_BROWSER_MANAGEMENT_URL': 'http://127.0.0.1:8320'})
    manager = Manager(directory, [sys.executable, str(Path(__file__).with_name('container_runtime.py')),
                                 sys.executable, str(Path(__file__).with_name('managed_adapter.py'))], env)
    server = ThreadingHTTPServer(('127.0.0.1', 8320), handler_for(manager, key))
    server.daemon_threads = True
    stopping = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.set())
    gateway = None
    try:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        gateway = subprocess.Popen(['/app/sub2api', *sys.argv[1:]], env=env)
        while not stopping.wait(1):
            if gateway.poll() is not None:
                break
            manager.tick()
    finally:
        server.shutdown()
        server.server_close()
        with manager.lock:
            manager.stop()
        if gateway is not None and gateway.poll() is None:
            gateway.terminate()
            try:
                gateway.wait(timeout=30)
            except subprocess.TimeoutExpired:
                gateway.kill()
                gateway.wait()
    if gateway is not None and gateway.returncode and not stopping.is_set():
        raise SystemExit(gateway.returncode)


if __name__ == '__main__':
    main()
