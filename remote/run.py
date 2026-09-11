"""Run bounded correctness/benchmark jobs and archive success or failure evidence."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from remote.bundle import build_bundle, source_hashes


def build_steps(args, output):
    python = sys.executable
    gpu = args.mode == 'gpu'
    if gpu:
        preflight = [python, str(ROOT / 'remote/check_cuda.py'), '--output', str(output / 'environment.json')]
        if args.expect_gpu:
            preflight.extend(['--expect-gpu', args.expect_gpu])
    else:
        preflight = [python, '-m', 'routekernel', 'doctor']
    steps = [
        ('preflight', preflight),
        ('routing-tests', [python, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_routing.py', '-v']),
        ('cpu-runtime-tests', [python, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_runtime.py', '-k', 'RuntimeTests', '-v']),
        ('numerical-gate', [python, str(ROOT / 'remote/validate.py'), '--device', 'cuda' if gpu else 'cpu',
                            '--output', str(output / 'correctness.json')]),
    ]
    if gpu:
        cases = json.loads((ROOT / 'remote/suites.json').read_text(encoding='utf-8'))[args.suite]
        backends = ['pytorch', 'grouped', 'torch-compiled', 'triton']
    else:
        cases = [{'name': 'cpu-smoke', 'tokens': 16, 'experts': 4, 'top_k': 2, 'dim': 7,
                  'hidden': 13, 'distribution': 'uniform', 'dtype': 'float32'}]
        backends = ['pytorch', 'grouped']
    for case in cases:
        command = [python, '-m', 'routekernel', 'bench', '--device', 'cuda' if gpu else 'cpu',
                   '--backends', *backends, '--warmup', str(args.warmup), '--repeats', str(args.repeats),
                   '--seed', str(args.seed), '--threads', '1', '--output', str(output / (case['name'] + '.json'))]
        for field in ('tokens', 'experts', 'top_k', 'dim', 'hidden', 'distribution', 'dtype'):
            command.extend(['--' + field.replace('_', '-'), str(case[field])])
        steps.append(('bench-' + case['name'], command))
    if args.profile:
        steps.append(('profile', [python, '-m', 'routekernel', 'profile', '--device', 'cuda' if gpu else 'cpu',
                                  '--backend', 'triton' if gpu else 'grouped', '--output', str(output / 'profile.json')]))
    return steps


def _kill_step(process):
    if os.name == 'posix':
        # Every step owns its process group, including compiler subprocesses.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        process.kill()
    process.wait()


def execute_step(command, log_path, timeout, env=None):
    start = time.monotonic()
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=(os.name == 'posix'))
        try:
            code = process.wait(timeout=timeout)
            status = 'passed' if code == 0 else 'failed'
        except subprocess.TimeoutExpired:
            _kill_step(process)
            code, status = None, 'timeout'
        except BaseException:
            _kill_step(process)
            raise
    return {'status': status, 'returncode': code, 'elapsed_seconds': time.monotonic() - start}


def write_summary(output, summary):
    (output / 'status.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    title = 'NVIDIA GPU run' if summary['mode'] == 'gpu' else 'CPU HARNESS REHEARSAL — NOT GPU VALIDATION'
    lines = [f'# {title}', '', f"Status: **{summary['status']}**", '',
             '| Step | Status | Seconds | Log |', '| --- | --- | ---: | --- |']
    for step in summary['steps']:
        lines.append(f"| {step['name']} | {step['status']} | {step['elapsed_seconds']:.2f} | [{step['name']}.log]({step['name']}.log) |")
    if summary.get('error'):
        lines.extend(['', 'Runner error: ' + summary['error']])
    lines.extend(['', '## Benchmark results', '',
                  '| Case | Backend | p50 ms | p95 ms | Speedup vs same-device PyTorch |',
                  '| --- | --- | ---: | ---: | ---: |'])
    for path in sorted(output.glob('*.json')):
        if path.name in ('status.json', 'environment.json', 'correctness.json', 'source-sha256.json', 'profile.json'):
            continue
        try:
            report = json.loads(path.read_text(encoding='utf-8'))
            for backend, result in report.get('results', {}).items():
                if result['status'] == 'ok':
                    relative = result.get('speedup_vs_pytorch')
                    label = f'{relative:.2f}x' if relative is not None else 'n/a'
                    lines.append(f"| [{path.stem}]({path.name}) | {backend} | {result['p50_ms']:.4f} | {result['p95_ms']:.4f} | {label} |")
                else:
                    lines.append(f"| [{path.stem}]({path.name}) | {backend} | ERROR | ERROR | n/a |")
        except (ValueError, KeyError):
            continue
    lines.extend(['', 'Supplied routes through merged output. Router scoring, validation, initial compilation,',
                  'and input generation are excluded. All compared backends run on the same device.',
                  'The compiled backend compiles expert arithmetic only. Results are synthetic single-layer',
                  'measurements, not full-model inference. Raw samples and timing scope are in each JSON.', '',
                  '**This script does not stop or terminate your rental. Download results, then terminate',
                  'the instance in the provider console when finished.**', ''])
    (output / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')


def archive_results(output):
    archive = output.with_name(output.name + '.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file() and not path.is_symlink():
                bundle.write(path, path.relative_to(output).as_posix())
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('gpu', 'cpu-smoke'), default='gpu')
    parser.add_argument('--suite', choices=('smoke', 'sweep'), default='smoke')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expect-gpu', help='require a GPU name substring, e.g. RTX 4090')
    parser.add_argument('--image-label', help='record the provider template/image tag or digest you selected')
    parser.add_argument('--max-seconds', type=int, default=1800, help='overall step execution deadline; does not stop rental billing')
    parser.add_argument('--step-timeout', type=int, default=600)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if min(args.max_seconds, args.step_timeout, args.repeats) < 1 or args.warmup < 0:
        parser.error('timeouts and repeats must be positive; warmup must be nonnegative')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output or ROOT / 'remote-results' / (args.mode + '-' + stamp)).resolve()
    steps = build_steps(args, output)
    if args.dry_run:
        print(json.dumps({'mode': args.mode, 'output': str(output), 'steps': steps,
                          'max_seconds': args.max_seconds, 'creates_paid_resources': False}, indent=2))
        return 0
    if output.exists() or output.with_name(output.name + '.zip').exists():
        parser.error('Output directory or archive already exists; choose a new path')
    output.mkdir(parents=True)
    summary = {'mode': args.mode, 'suite': args.suite if args.mode == 'gpu' else 'cpu-smoke',
               'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
               'image_label_unverified': args.image_label, 'steps': [],
               'max_seconds': args.max_seconds, 'step_timeout': args.step_timeout}
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    # No optional hardware tests are disguised as passed in this harness:
    # remote/validate.py explicitly invokes every required backend instead.
    start = time.monotonic()
    try:
        hashes = source_hashes()
        (output / 'source-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
        build_bundle(output / 'source.tar.gz')
        for name, command in steps:
            remaining = args.max_seconds - (time.monotonic() - start)
            if remaining <= 0:
                summary['status'] = 'timeout'
                break
            print(f'Running {name} ... log: {output / (name + ".log")}', flush=True)
            result = execute_step(command, output / (name + '.log'), min(args.step_timeout, remaining), env)
            summary['steps'].append({'name': name, 'command': command, **result})
            write_summary(output, summary)
            print(f"{name}: {result['status']}", flush=True)
            if result['status'] != 'passed':
                summary['status'] = result['status']
                break
        else:
            summary['status'] = 'passed'
    except KeyboardInterrupt:
        summary['status'] = 'interrupted'
    except Exception as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        summary['elapsed_seconds'] = time.monotonic() - start
        write_summary(output, summary)
        archive = archive_results(output)
        print(f"Run {summary['status']}. Download: {archive}")
        print('If using a rental, billing continues until you stop/terminate it in the provider console.')
    return int(summary['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
