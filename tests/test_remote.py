import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from remote.bundle import build_bundle
from remote.run import ROOT, build_steps, execute_step, main
from remote.setup import setup_commands


class RemoteHarnessTests(unittest.TestCase):
    def test_bundle_captures_uncommitted_source_excludes_environment_and_git(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, content in {
                'README.md': '# Example', 'routekernel/new_file.py': 'x = 1',
                'routekernel/shaders/a.wgsl': '// shader', 'remote/run.py': '# runner',
                '.env': 'SECRET=value', '.git/config': 'credential',
                '.venv/lib/code.py': '# installed dependency', 'private.pem': 'secret',
                'remote/credentials.json': '{"token": "secret"}',
                'routekernel/__pycache__/cache.py': '# generated',
            }.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            first, second = root / 'one.tar.gz', root / 'two.tar.gz'
            manifest = build_bundle(first, root)
            build_bundle(second, root)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertIn('routekernel/new_file.py', manifest)
            self.assertNotIn('.env', manifest)
            self.assertNotIn('remote/credentials.json', manifest)
            with tarfile.open(first) as archive:
                names = archive.getnames()
                self.assertTrue(all(name.startswith('routekernel-remote/') for name in names))
                self.assertFalse(any('.venv' in name or '.git/' in name or 'private.pem' in name for name in names))
                self.assertIn('routekernel-remote/BUNDLE-MANIFEST.json', names)
            with self.assertRaises(FileExistsError):
                build_bundle(first, root)

    def test_gpu_plan_gates_all_backends_before_benchmarking(self):
        args = argparse.Namespace(mode='gpu', suite='smoke', expect_gpu='4090', warmup=2, repeats=3, seed=0, profile=True)
        steps = build_steps(args, Path('/tmp/results'))
        self.assertEqual([name for name, _ in steps[:4]], ['preflight', 'routing-tests', 'cpu-runtime-tests', 'numerical-gate'])
        self.assertIn('4090', steps[0][1])
        for name, command in steps:
            if name.startswith('bench-'):
                self.assertEqual(command[command.index('--device') + 1], 'cuda')
                self.assertIn('triton', command)
                self.assertIn('torch-compiled', command)
        self.assertEqual(steps[-1][0], 'profile')

    def test_setup_preserves_template_torch_and_fresh_uses_official_index(self):
        reuse = setup_commands('reuse', 'python3', Path('/work/venv'), '2.13.0', 'cu126')
        self.assertTrue(reuse[0][-1].endswith('check_cuda.py'))
        self.assertIn('--system-site-packages', reuse[1])
        self.assertFalse(any('torch==2.13.0' in command for command in reuse))
        fresh = setup_commands('fresh', 'python3', Path('/work/venv'), '2.13.0', 'cu126')
        self.assertNotIn('--system-site-packages', fresh[0])
        self.assertIn('https://download.pytorch.org/whl/cu126', fresh[1])

    def test_timeout_kills_step_and_keeps_log(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'timeout.log'
            result = execute_step([sys.executable, '-u', '-c', 'import time; print("started", flush=True); time.sleep(30)'], log, 1)
            self.assertEqual(result['status'], 'timeout')
            self.assertIn('started', log.read_text())
            self.assertLess(result['elapsed_seconds'], 10)

    def test_failure_stops_later_steps_and_archives_evidence(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            output = Path(folder) / 'failure'
            def fail(command, path, timeout, env):
                path.write_text('GPU check failed\n')
                return {'status': 'failed', 'returncode': 1, 'elapsed_seconds': 0.1}
            with patch('remote.run.execute_step', side_effect=fail) as execute:
                self.assertEqual(main(['--output', str(output)]), 1)
                self.assertEqual(execute.call_count, 1)
            status = json.loads((output / 'status.json').read_text())
            self.assertEqual(status['status'], 'failed')
            with zipfile.ZipFile(output.with_suffix('.zip')) as archive:
                self.assertIn('preflight.log', archive.namelist())
                self.assertIn('source.tar.gz', archive.namelist())
                self.assertIn('SUMMARY.md', archive.namelist())
                self.assertNotIn('correctness.json', archive.namelist())

    def test_dry_run_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as out:
            target = Path(folder) / 'dry'
            self.assertEqual(main(['--dry-run', '--output', str(target)]), 0)
            self.assertFalse(target.exists())
            self.assertFalse(json.loads(out.getvalue())['creates_paid_resources'])

    def test_cpu_rehearsal_runs_end_to_end_and_is_not_labelled_gpu_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'cpu-rehearsal'
            result = subprocess.run([sys.executable, str(ROOT / 'remote/run.py'), '--mode', 'cpu-smoke',
                                     '--warmup', '0', '--repeats', '2', '--output', str(output)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            status = json.loads((output / 'status.json').read_text())
            self.assertEqual(status['mode'], 'cpu-smoke')
            self.assertEqual(status['status'], 'passed')
            self.assertIn('NOT GPU VALIDATION', (output / 'SUMMARY.md').read_text())
            with zipfile.ZipFile(output.with_suffix('.zip')) as archive:
                self.assertIn('cpu-smoke.json', archive.namelist())
                self.assertIn('correctness.json', archive.namelist())


if __name__ == '__main__':
    unittest.main()
