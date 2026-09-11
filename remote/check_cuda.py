"""Fail closed before paid GPU experiments; always print a diagnostic report."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import os
import shutil
import subprocess
import sys


def inspect_environment(expect_gpu=None):
    report = {'python': platform.python_version(), 'platform': platform.platform(), 'errors': []}
    errors = report['errors']
    if sys.platform != 'linux':
        errors.append('Linux is required for this Triton harness')
    if sys.version_info < (3, 10):
        errors.append('Python 3.10 or newer is required')
    report['compiler'] = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
    if not report['compiler']:
        errors.append('A C++ compiler is required for the compiled backend; choose a development image')
    report['diagnostic_environment'] = {key: os.environ[key] for key in (
        'CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING', 'TRITON_INTERPRET',
        'TORCHDYNAMO_DISABLE', 'TORCH_COMPILE_DISABLE',
    ) if key in os.environ}
    for key in ('CUDA_LAUNCH_BLOCKING', 'TRITON_INTERPRET', 'TORCHDYNAMO_DISABLE', 'TORCH_COMPILE_DISABLE'):
        if os.environ.get(key, '').lower() not in ('', '0', 'false'):
            errors.append(f'Unset {key} before benchmarking genuine GPU/compiled execution')
    try:
        import torch
        report.update(pytorch=torch.__version__, cuda_build=torch.version.cuda,
                      cuda_available=torch.cuda.is_available())
        if tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) < (2, 5):
            errors.append('PyTorch >= 2.5 is required')
        if torch.version.cuda is None or not torch.cuda.is_available():
            errors.append('CUDA-enabled PyTorch and a visible NVIDIA GPU are required; CPU/ROCm builds do not qualify')
        else:
            props = torch.cuda.get_device_properties(0)
            report.update(gpu=props.name, compute_capability=[props.major, props.minor],
                          gpu_memory_bytes=props.total_memory, visible_gpu_count=torch.cuda.device_count())
            if props.major < 8:
                errors.append('This harness targets NVIDIA compute capability 8.0 or newer')
            if expect_gpu and expect_gpu.lower() not in props.name.lower():
                errors.append(f'Expected GPU containing {expect_gpu!r}, found {props.name!r}')
            result = torch.tensor([[1., 2.]], device='cuda') @ torch.tensor([[3.], [4.]], device='cuda')
            torch.cuda.synchronize()
            report['cuda_matmul_check'] = result.item() == 11.0
            if not report['cuda_matmul_check']:
                errors.append('CUDA arithmetic check failed')
            report['bf16_supported'] = torch.cuda.is_bf16_supported()
    except Exception as exc:
        errors.append(f'PyTorch/CUDA check: {type(exc).__name__}: {exc}')
    try:
        import triton
        report['triton'] = triton.__version__
        if tuple(int(x) for x in triton.__version__.split('.')[:2]) < (3, 1):
            errors.append('Triton >= 3.1 is required')
    except Exception as exc:
        errors.append(f'Triton import: {type(exc).__name__}: {exc}')
    try:
        import numpy
        report['numpy'] = numpy.__version__
    except ImportError:
        errors.append('NumPy is required')
    executable = shutil.which('nvidia-smi')
    if executable:
        try:
            result = subprocess.run([executable, '--query-gpu=name,uuid,driver_version,memory.total,memory.used,temperature.gpu,utilization.gpu,power.draw', '--format=csv'], capture_output=True, text=True, timeout=15)
            report['nvidia_smi'] = result.stdout + result.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:
            report['nvidia_smi'] = str(exc)
    # Names and versions only: no environment dump, package URLs, or credentials.
    report['packages'] = sorted(
        [{'name': dist.metadata['Name'], 'version': dist.version} for dist in importlib.metadata.distributions()],
        key=lambda item: item['name'].lower(),
    )
    report['status'] = 'failed' if errors else 'passed'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expect-gpu')
    args = parser.parse_args(argv)
    report = inspect_environment(args.expect_gpu)
    content = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(content, encoding='utf-8')
    print(content)
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
