"""Mandatory numerical gate: explicit backends, with no hardware-test skips."""

import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    import torch
    from routekernel.benchmark import tolerances
    from routekernel.runtime import ExpertWeights, forward, reference_forward
    from routekernel.workloads import make_workload
    report = {'status': 'running', 'device': args.device, 'reference_device': 'cpu', 'cases': []}
    torch.set_num_threads(1)
    backends = ('pytorch', 'grouped') if args.device == 'cpu' else ('pytorch', 'grouped', 'torch-compiled', 'triton')
    dtypes = [torch.float32] if args.device == 'cpu' else [torch.float32, torch.float16]
    if args.device == 'cuda' and torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    report['backends'] = backends
    report['dtypes'] = [str(dtype) for dtype in dtypes]
    shapes = [(0, 5, 2, 7, 13), (1, 1, 1, 7, 13), (17, 5, 2, 37, 71), (65, 5, 5, 19, 33)]
    try:
        with torch.inference_mode():
            for dtype in dtypes:
                for shape in shapes:
                    for distribution in ('uniform', 'skewed'):
                        cpu = make_workload(*shape, distribution=distribution, seed=3, dtype=dtype)
                        expected = reference_forward(*cpu)
                        x, ids, weights, experts = cpu
                        inputs = (x.to(args.device), ids.to(args.device), weights.to(args.device),
                                  ExpertWeights(experts.up.to(args.device), experts.down.to(args.device)))
                        for backend in backends:
                            case = {'shape': shape, 'distribution': distribution, 'dtype': str(dtype), 'backend': backend, 'status': 'running'}
                            report['cases'].append(case)
                            actual = forward(*inputs, backend).cpu()
                            torch.testing.assert_close(actual, expected, **tolerances(dtype))
                            case.update(status='passed', max_abs_error=(actual.float() - expected.float()).abs().max().item() if actual.numel() else 0.0)
                            print(f'PASS {backend} {dtype} {shape} {distribution}', flush=True)
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        if report['cases']:
            report['cases'][-1]['status'] = 'failed'
        traceback.print_exc()
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
