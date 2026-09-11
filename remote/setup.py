"""Create an isolated remote environment without changing GPU drivers."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def setup_commands(mode, python, env_dir, torch_version, cuda_wheel):
    env_python = str(env_dir / 'bin' / 'python')
    create = [python, '-m', 'venv']
    if mode == 'reuse':
        create.append('--system-site-packages')
    create.append(str(env_dir))
    commands = []
    if mode == 'reuse':
        commands.append([python, str(ROOT / 'remote/check_cuda.py')])
    commands.append(create)
    if mode == 'fresh':
        commands.append([env_python, '-m', 'pip', 'install', '--no-input',
                         f'torch=={torch_version}', '--index-url', f'https://download.pytorch.org/whl/{cuda_wheel}'])
    commands.extend([
        [env_python, '-m', 'pip', 'install', '--no-input', 'numpy>=1.26', 'setuptools>=68', 'wheel'],
        [env_python, '-m', 'pip', 'install', '--no-input', '--no-deps', '--no-build-isolation', '-e', str(ROOT)],
        [env_python, '-m', 'pip', 'check'],
        [env_python, str(ROOT / 'remote' / 'check_cuda.py')],
    ])
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('reuse', 'fresh'), default='reuse',
                        help='reuse CUDA PyTorch/Triton from a prepared template, or install a fresh CUDA wheel')
    parser.add_argument('--torch-version', default='2.13.0', help='fresh mode only')
    parser.add_argument('--cuda-wheel', default='cu126', help='fresh mode only, official PyTorch wheel channel')
    parser.add_argument('--dry-run', action='store_true', help='print the recipe without creating or installing anything')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'\d+\.\d+\.\d+', args.torch_version) or not re.fullmatch(r'cu\d{3}', args.cuda_wheel):
        parser.error('Use a numeric torch version and CUDA wheel channel such as cu126')
    env_dir = ROOT / '.venv-remote'
    commands = setup_commands(args.mode, sys.executable, env_dir, args.torch_version, args.cuda_wheel)
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        return 0
    if sys.platform != 'linux':
        parser.exit(1, 'Remote setup requires Linux. Use --dry-run to inspect the recipe locally.\n')
    if env_dir.exists():
        parser.exit(1, '.venv-remote already exists; reuse it or choose a fresh extraction directory.\n')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log_dir = ROOT / 'remote-results' / ('setup-' + stamp)
    log_dir.mkdir(parents=True)
    if args.mode == 'reuse':
        # Preserve diagnostics even when the image fails before venv creation.
        commands[0].extend(['--output', str(log_dir / 'preflight.json')])
    (log_dir / 'recipe.json').write_text(json.dumps(commands, indent=2) + '\n', encoding='utf-8')
    with (log_dir / 'setup.log').open('w', encoding='utf-8') as log:
        for command in commands:
            print(shlex.join(command), flush=True)
            log.write(shlex.join(command) + '\n')
            log.flush()
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print(f'Setup failed. Inspect {log_dir / "setup.log"}', file=sys.stderr)
                return result.returncode
    print('Ready: .venv-remote/bin/python remote/run.py --suite smoke')
    print(f'Setup log: {log_dir / "setup.log"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
