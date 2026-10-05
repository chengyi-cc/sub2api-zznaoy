import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import yaml

spec = importlib.util.spec_from_file_location('upgrade', Path(__file__).with_name('upgrade_config.py'))
upgrade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upgrade)


class UpgradeTest(unittest.TestCase):
    def source(self):
        return {'name': 'second-instance', 'services': {
            'sub2api': {'image': 'example:old', 'environment': ['DATABASE_PASSWORD=${POSTGRES_PASSWORD:?required}', 'JWT_SECRET=${JWT_SECRET}', 'LITERAL=$$abc'],
                        'volumes': ['./data:/app/data'], 'ports': ['18080:8080'],
                        'security_opt': ['no-new-privileges:true']},
            'postgres': {'image': 'postgres:18-alpine', 'volumes': ['./postgres_data:/var/lib/postgresql/data']}},
                'networks': {'default': {'name': 'original-network'}}}

    def resolved(self):
        config = self.source()
        app = config['services']['sub2api']
        app['environment'] = {'DATABASE_PASSWORD': 'secret', 'JWT_SECRET': 'token', 'LITERAL': '$abc'}
        app['volumes'] = [{'type': 'bind', 'source': '/srv/second/data', 'target': '/app/data'}]
        return config

    def live(self):
        return {'Config': {'Env': ['DATABASE_PASSWORD=secret', 'JWT_SECRET=token', 'LITERAL=$abc']},
                'Mounts': [{'Source': '/srv/second/data', 'Destination': '/app/data', 'RW': True}]}

    def test_preserves_other_services_and_environment_expressions(self):
        original = self.source()
        result = upgrade.transform(original, 'example:new', './prism/seccomp-sub2api.json')
        self.assertEqual(original, self.source())
        self.assertEqual(result['services']['postgres'], original['services']['postgres'])
        self.assertEqual(result['services']['sub2api']['environment'], original['services']['sub2api']['environment'])
        self.assertEqual(yaml.safe_load(yaml.safe_dump(result)), result)
        self.assertEqual(result['services']['sub2api']['security_opt'], ['seccomp=./prism/seccomp-sub2api.json'])
        self.assertNotIn('privileged', result['services']['sub2api'])

    def test_custom_runtime_policy_is_not_overwritten(self):
        cases = [('entrypoint', '/custom'), ('command', ['--custom']), ('privileged', True),
                 ('user', 'nobody'), ('network_mode', 'host'), ('read_only', True),
                 ('security_opt', ['seccomp=/custom/profile.json']), ('extends', {'file': 'elsewhere.yml'})]
        for key, value in cases:
            with self.subTest(key=key):
                config = self.source()
                config['services']['sub2api'][key] = value
                with self.assertRaises(ValueError):
                    upgrade.transform(config, 'example:new', './prism/seccomp-sub2api.json')

    def test_resolved_config_data_and_live_environment_must_match(self):
        before = self.resolved()
        after = upgrade.transform(before, 'example:new', './prism/seccomp-sub2api.json')
        upgrade.validate(before, after, self.live())
        for field in ['DATABASE_PASSWORD', 'JWT_SECRET']:
            live = self.live()
            live['Config']['Env'] = [entry for entry in live['Config']['Env'] if not entry.startswith(field + '=')]
            with self.assertRaisesRegex(ValueError, 'environment differs'):
                upgrade.validate(before, after, live)
        live = self.live()
        live['Mounts'][0]['Source'] = '/srv/first/data'
        with self.assertRaisesRegex(ValueError, 'data mount differs'):
            upgrade.validate(before, after, live)
        after['services']['postgres']['image'] = 'postgres:other'
        with self.assertRaisesRegex(ValueError, 'outside the permitted'):
            upgrade.validate(before, after, self.live())

    def test_named_volumes_keep_original_project(self):
        before = self.resolved()
        before['services']['sub2api']['volumes'][0] = {'type': 'volume', 'source': 'data', 'target': '/app/data'}
        before['volumes'] = {'data': {'name': 'second_data'}}
        live = self.live()
        live['Mounts'][0] = {'Name': 'second_data', 'Destination': '/app/data', 'RW': True}
        upgrade.validate(before, upgrade.transform(before, 'example:new', './prism/seccomp-sub2api.json'), live)
        live['Mounts'][0]['Name'] = 'first_data'
        with self.assertRaises(ValueError):
            upgrade.validate(before, upgrade.transform(before, 'example:new', './prism/seccomp-sub2api.json'), live)

    def test_prepare_does_not_change_source_and_install_keeps_backup(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / 'docker-compose.local.yml'
            contents = '# original comment\n' + yaml.safe_dump(self.source())
            source.write_text(contents, encoding='utf-8')
            output = io.StringIO()
            with redirect_stdout(output):
                upgrade.prepare(directory, source.name, 'example:new')
            backup = output.getvalue().strip()
            self.assertEqual(source.read_text(encoding='utf-8'), contents)
            upgrade.install(directory, source.name, backup)
            self.assertEqual((directory / backup / 'original.yml').read_text(encoding='utf-8'), contents)
            self.assertEqual(yaml.safe_load(source.read_text(encoding='utf-8'))['services']['sub2api']['image'], 'example:new')
            self.assertTrue((directory / 'prism/seccomp-sub2api.json').is_file())
            with self.assertRaisesRegex(ValueError, 'changed during upgrade'):
                upgrade.install(directory, source.name, backup)

    def test_missing_persistent_data_is_rejected(self):
        before = self.resolved()
        before['services']['sub2api']['volumes'] = []
        with self.assertRaisesRegex(ValueError, 'persistent'):
            upgrade.validate(before, upgrade.transform(before, 'example:new', 'profile'), self.live())


if __name__ == '__main__':
    unittest.main()
