"""Measure AMD-capable custom shaders with explicit transfer boundaries."""

from datetime import datetime, timezone
import platform

import torch

from .benchmark import measure, tolerances
from .runtime import _execute, reference_forward, validate_inputs
from .wgpu_backend import WgpuEngine, _wgpu
from .workloads import make_workload


@torch.inference_mode()
def run_wgpu_benchmark(
    *, num_tokens=128, num_experts=8, top_k=2, dim=64, hidden=128,
    distribution='uniform', seed=0, warmup=5, repeats=25, threads=1, adapter_name=None,
):
    if num_tokens < 1 or repeats < 1 or warmup < 0 or threads < 1:
        raise ValueError('tokens, repeats and threads must be positive; warmup must be nonnegative')
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        inputs = make_workload(num_tokens, num_experts, top_k, dim, hidden, distribution, seed)
        validate_inputs(*inputs)
        expected = reference_forward(*inputs)
        tokens, ids, weights, experts = inputs
        results = {}

        def checked_measure(name, call, scope):
            actual = call()
            torch.testing.assert_close(actual, expected, **tolerances(torch.float32))
            result = measure(call, torch.device('cpu'), warmup, repeats)
            result.update({
                'status': 'ok', 'correctness': 'passed', 'scope': scope,
                'max_abs_error': (actual - expected).abs().max().item(),
                'tokens_per_second': num_tokens * 1000 / result['p50_ms'],
                'assignments_per_second': ids.numel() * 1000 / result['p50_ms'],
            })
            results[name] = result

        for backend in ('pytorch', 'grouped'):
            checked_measure('cpu-' + backend, lambda: _execute(*inputs, backend),
                            'CPU routes through merged CPU output; validation excluded')

        with WgpuEngine(experts, adapter_name=adapter_name) as engine:
            def request():
                with engine.prepare(tokens, ids, weights, validate=False) as plan:
                    return plan.run()

            checked_measure('wgpu-request', request,
                            'CPU route compilation and tile scheduling + input/metadata upload + GPU gather/MLP/merge + full output readback + per-request buffer lifecycle; expert weights resident')
            with engine.prepare(tokens, ids, weights, validate=False) as plan:
                actual = plan.run()
                torch.testing.assert_close(actual, expected, **tolerances(torch.float32))

                def resident():
                    plan.submit()
                    plan.synchronize()

                result = measure(resident, torch.device('cpu'), warmup, repeats)
                result.update({
                    'status': 'ok', 'correctness': 'passed',
                    'scope': 'CPU command encoding + GPU gather/MLP/merge + four-byte completion read; inputs/metadata/weights already resident; no full output readback',
                    'max_abs_error': (actual - expected).abs().max().item(),
                    'tokens_per_second': num_tokens * 1000 / result['p50_ms'],
                    'assignments_per_second': ids.numel() * 1000 / result['p50_ms'],
                })
                results['wgpu-resident'] = result
            info = engine.context.info
        # Only compare the full-request path to CPU; resident excludes transfers.
        results['wgpu-request']['speedup_vs_cpu_pytorch'] = results['cpu-pytorch']['p50_ms'] / results['wgpu-request']['p50_ms']
        results['wgpu-request']['speedup_vs_cpu_grouped'] = results['cpu-grouped']['p50_ms'] / results['wgpu-request']['p50_ms']
        counts = torch.bincount(ids.flatten(), minlength=num_experts).double()
        return {
            'schema_version': 1, 'benchmark_kind': 'wgpu-cpu-boundary',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'workload': {'source': 'synthetic', 'tokens': num_tokens, 'experts': num_experts,
                         'top_k': top_k, 'dim': dim, 'hidden': hidden, 'dtype': 'float32',
                         'distribution': distribution, 'seed': seed,
                         'expert_architecture': 'bias-free Linear-ReLU-Linear'},
            'environment': {'python': platform.python_version(), 'pytorch': torch.__version__,
                            'wgpu': _wgpu().__version__, 'platform': platform.platform(),
                            'processor': platform.processor(), 'cpu_threads': threads,
                            'gpu_adapter': info, 'software_adapter_allowed': False},
            'measurement': {'warmup': warmup, 'repeats': repeats,
                            'timer': 'perf_counter_ns; request waits on full output readback; resident waits on a blocking four-byte output read',
                            'excluded_from_all': ['input generation', 'validation', 'router scoring', 'reference checks', 'shader compilation', 'initial expert-weight upload'],
                            'input_reuse': 'same synthetic tensors reused; weights stay resident on GPU',
                            'precision': 'FP32 shader arithmetic and accumulation',
                            'resident_is_kernel_only': False, 'tolerances': tolerances(torch.float32),
                            'backend_order': list(results)},
            'routing': {'expert_counts': counts.int().tolist(), 'assignments': ids.numel(),
                        'min_load': int(counts.min()), 'max_load': int(counts.max()),
                        'coefficient_of_variation': (counts.std(correction=0) / counts.mean()).item()},
            'results': results,
        }
    finally:
        torch.set_num_threads(old_threads)
