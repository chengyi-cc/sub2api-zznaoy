import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import Mock, patch

from managed_runtime import Manager, bridge_key, handler_for
import managed_adapter


class ManagedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.manager = Manager(self.directory, [sys.executable, '-c', 'pass'], os.environ.copy())
        self.addCleanup(self.manager.stop)

    def test_secret_is_stable_private_and_never_in_status(self):
        key = bridge_key(self.directory)
        self.assertEqual(key, bridge_key(self.directory))
        self.assertGreaterEqual(len(key), 32)
        self.assertNotIn(key, json.dumps(self.manager.snapshot(True)))
        if os.name == 'posix':
            self.assertEqual(0o600, (self.directory / 'bridge.key').stat().st_mode & 0o777)

    def test_live_start_stop_restart_and_persisted_intent(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        # A local fixture, no real browser, credentials or upstream request.
        script = '''from http.server import HTTPServer, BaseHTTPRequestHandler
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
 def log_message(self,*args): pass
HTTPServer(('127.0.0.1', PORT), H).serve_forever()
'''.replace('PORT', str(port))
        self.manager.command = [sys.executable, '-u', '-c', script]
        self.manager.adapter_port = port
        self.manager.action('start')
        for _ in range(50):
            self.manager.tick()
            if self.manager.state == 'running':
                break
            time.sleep(0.02)
        self.assertEqual('running', self.manager.state)
        first = self.manager.process.pid
        self.manager.action('start')
        self.assertEqual(first, self.manager.process.pid, 'start must be idempotent')
        restored = Manager(self.directory, [], {}, port)
        self.assertTrue(restored.desired)
        self.manager.action('restart')
        self.assertNotEqual(first, self.manager.process.pid)
        self.manager.action('stop')
        self.assertFalse(self.manager.healthy())
        self.assertEqual('stopped', self.manager.state)
        self.assertFalse(Manager(self.directory, [], {}, port).desired)

    def test_stop_preserves_unknown_turn_state(self):
        marker = self.directory / 'pending.json'
        marker.write_text('{"pending":true}', encoding='utf-8')
        self.manager.action('stop')
        self.assertEqual('{"pending":true}', marker.read_text(encoding='utf-8'))

    def test_managed_stop_runs_upstream_cleanup(self):
        cleaned = []
        def upstream_main():
            try:
                managed_adapter.stop()
            finally:
                cleaned.append(True)
        with patch.dict(sys.modules, {'server': Mock(main=upstream_main)}), patch('managed_adapter.signal.signal') as register:
            managed_adapter.main()
            register.assert_called_once_with(managed_adapter.signal.SIGTERM, managed_adapter.stop)
        self.assertEqual([True], cleaned)

    def test_unrelated_service_not_claimed(self):
        with patch.object(self.manager, 'healthy', return_value=True), patch('managed_runtime.subprocess.Popen') as spawn:
            self.manager.action('start')
            spawn.assert_not_called()
            self.assertEqual('error', self.manager.state)

    def test_failed_persistence_does_not_stop_service(self):
        with patch('managed_runtime.atomic_json', side_effect=OSError('secret-path')), patch.object(self.manager, 'stop') as stop:
            with self.assertRaises(OSError):
                self.manager.action('stop')
            stop.assert_not_called()

    def test_only_allowlisted_events_leave_subprocess(self):
        process = Mock(stdout=io.BytesIO(
            b'Bearer SECRET\n'
            b'{"event":"prism_adapter_error","message":"SECRET","class":"SECRET"}\n'
            b'{"event":"SECRET"}\n' + b'x'*5000 + b'\n'))
        self.manager.capture(process)
        text = json.dumps(self.manager.snapshot(True))
        self.assertNotIn('SECRET', text)
        self.assertIn('prism_adapter_error', text)
        for _ in range(300): self.manager.event('check_ok')
        self.assertEqual(200, len(self.manager.logs))

    def test_management_auth_and_operation_allowlist(self):
        key = 'k' * 48
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(self.manager, key))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{server.server_port}'
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for path, headers, method, expected in [
            ('/status', {}, 'GET', 401),
            ('/stop', {'Authorization': 'Bearer wrong'}, 'POST', 401),
            ('/stop', {'Authorization': 'Bearer '+key, 'Origin': 'https://evil.test'}, 'POST', 401),
            ('/shell', {'Authorization': 'Bearer '+key}, 'POST', 404),
        ]:
            req = urllib.request.Request(base+path, headers=headers, method=method)
            with self.assertRaises(urllib.error.HTTPError) as caught:
                opener.open(req, timeout=2)
            self.assertEqual(expected, caught.exception.code)
        req = urllib.request.Request(base+'/status', headers={'Authorization': 'Bearer '+key})
        with opener.open(req, timeout=2) as result:
            self.assertEqual('stopped', json.load(result)['state'])


if __name__ == '__main__':
    unittest.main()
