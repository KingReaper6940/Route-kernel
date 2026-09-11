"""Correctness-gated, reproducible synthetic MoE latency measurements."""

from datetime import datetime, timezone
import importlib.metadata
import math
import platform
import statistics
import time

import torch

from .runtime import _execute, reference_forward, validate_inputs
from .workloads import make_workload


def percentile(samples: list[float], quantile: float) -> float:
    """Linearly interpolated sample percentile, including small sample sets."""
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def measure(call, device, warmup: int, repeats: int) -> dict:
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        call()
    synchronize()
    samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter_ns()
        call()
        synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "samples_ms": samples,
    }


def tolerances(dtype):
    if dtype == torch.bfloat16:
        return {"rtol": 0.03, "atol": 0.03}
    if dtype == torch.float16:
        return {"rtol": 0.005, "atol": 0.005}
    return {"rtol": 1e-4, "atol": 1e-5}


@torch.inference_mode()
def run_benchmark(
    *, num_tokens=128, num_experts=8, top_k=2, dim=64, hidden=128,
    distribution="uniform", seed=0, device="cpu", dtype="float32",
    backends=("pytorch", "grouped"), warmup=5, repeats=25, threads=1,
) -> dict:
    if num_tokens < 1 or repeats < 1 or warmup < 0 or threads < 1:
        raise ValueError("tokens, repeats and threads must be positive; warmup must be nonnegative")
    if not backends or len(set(backends)) != len(backends):
        raise ValueError("provide at least one backend, without duplicates")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype not in dtype_map:
        raise ValueError(f"dtype must be one of {tuple(dtype_map)}")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("benchmark device must be cpu or cuda")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch environment")
    original_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        inputs = make_workload(
            num_tokens, num_experts, top_k, dim, hidden, distribution, seed, target, dtype_map[dtype]
        )
        tokens, ids, weights, experts = inputs
        validate_inputs(*inputs)
        # The oracle runs outside timed regions and is never the speedup baseline.
        expected = reference_forward(*inputs)
        counts = torch.bincount(ids.flatten(), minlength=num_experts).cpu().double()
        report = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "workload": {
                "source": "synthetic", "tokens": num_tokens, "experts": num_experts,
                "top_k": top_k, "dim": dim, "hidden": hidden, "distribution": distribution,
                "seed": seed, "dtype": dtype, "expert_architecture": "bias-free Linear-ReLU-Linear",
            },
            "environment": {
                "python": platform.python_version(), "pytorch": torch.__version__,
                "platform": platform.platform(), "processor": platform.processor(),
                "device": str(target), "device_name": torch.cuda.get_device_name(target) if target.type == "cuda" else platform.processor(),
                "cuda_version": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            "measurement": {
                "warmup": warmup, "repeats": repeats,
                "timer": "perf_counter_ns; CUDA synchronized before and after each sample",
                "scope": "supplied routes to merged outputs; includes grouping, gather, expert execution and merge as applicable",
                "excluded": ["router scoring", "input validation", "workload creation", "correctness checks", "initial compilation"],
                "input_reuse": "same tensors reused; this is a steady-state microbenchmark",
                "compiled_scope": "expert MLP only; routing, expert loop and merge remain eager",
            },
            "routing": {
                "assignments": ids.numel(), "expert_counts": counts.int().tolist(),
                "min_load": int(counts.min()), "max_load": int(counts.max()),
                "coefficient_of_variation": (counts.std(correction=0) / counts.mean()).item(),
            },
            "results": {},
        }
        try:
            report["environment"]["triton"] = importlib.metadata.version("triton")
        except importlib.metadata.PackageNotFoundError:
            report["environment"]["triton"] = None
        for backend in backends:
            try:
                call = lambda: _execute(tokens, ids, weights, experts, backend)
                actual = call()  # First compilation, when applicable, is outside timing.
                torch.testing.assert_close(actual, expected, **tolerances(tokens.dtype))
                result = measure(call, target, warmup, repeats)
                result.update({
                    "status": "ok", "correctness": "passed",
                    "max_abs_error": (actual.float() - expected.float()).abs().max().item(),
                    "tolerances": tolerances(tokens.dtype),
                    "tokens_per_second": num_tokens * 1000 / result["p50_ms"],
                    "assignments_per_second": ids.numel() * 1000 / result["p50_ms"],
                })
                report["results"][backend] = result
            except (RuntimeError, ValueError, ImportError, AssertionError) as exc:
                report["results"][backend] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        baseline = report["results"].get("pytorch", {})
        if baseline.get("status") == "ok":
            for result in report["results"].values():
                if result["status"] == "ok":
                    result["speedup_vs_pytorch"] = baseline["p50_ms"] / result["p50_ms"]
        return report
    finally:
        torch.set_num_threads(original_threads)


@torch.inference_mode()
def profile_workload(output_path, *, backend="grouped", device="cpu"):
    """Export one warmed-up synthetic forward pass as a Chrome trace."""
    target = torch.device(device)
    inputs = make_workload(device=target)
    validate_inputs(*inputs)
    for _ in range(3):
        _execute(*inputs, backend)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if target.type == "cuda":
        torch.cuda.synchronize(target)
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
        with torch.profiler.record_function(f"routekernel_{backend}"):
            _execute(*inputs, backend)
        if target.type == "cuda":
            torch.cuda.synchronize(target)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(output_path))
