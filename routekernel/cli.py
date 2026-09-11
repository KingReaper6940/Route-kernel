"""Command-line demos, environment checks, benchmarks, and profiling."""

import argparse
import json
from pathlib import Path
import platform
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="routekernel", description="Synthetic sparse MoE execution experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="run the hand-checkable expert example")
    doctor = commands.add_parser("doctor", help="report the current Python/PyTorch/GPU environment")
    doctor.add_argument("--wgpu", action="store_true", help="enumerate native Vulkan/DirectX hardware adapters")
    gpu_bench = commands.add_parser("gpu-bench", help="benchmark custom WGSL shaders on a hardware GPU")
    gpu_bench.add_argument("--tokens", type=int, default=128)
    gpu_bench.add_argument("--experts", type=int, default=8)
    gpu_bench.add_argument("--top-k", type=int, default=2)
    gpu_bench.add_argument("--dim", type=int, default=64)
    gpu_bench.add_argument("--hidden", type=int, default=128)
    gpu_bench.add_argument("--distribution", choices=("uniform", "balanced", "skewed"), default="uniform")
    gpu_bench.add_argument("--seed", type=int, default=0)
    gpu_bench.add_argument("--warmup", type=int, default=5)
    gpu_bench.add_argument("--repeats", type=int, default=25)
    gpu_bench.add_argument("--threads", type=int, default=1)
    gpu_bench.add_argument("--adapter", help="GPU name substring, e.g. 'RX 6800'; software adapters are rejected")
    gpu_bench.add_argument("--output", type=Path)
    bench = commands.add_parser("bench", help="correctness-gated steady-state latency benchmark")
    bench.add_argument("--tokens", type=int, default=128)
    bench.add_argument("--experts", type=int, default=8)
    bench.add_argument("--top-k", type=int, default=2)
    bench.add_argument("--dim", type=int, default=64)
    bench.add_argument("--hidden", type=int, default=128)
    bench.add_argument("--distribution", choices=("uniform", "balanced", "skewed"), default="uniform")
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    bench.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    bench.add_argument("--backends", nargs="+", choices=("pytorch", "grouped", "torch-compiled", "triton"), default=["pytorch", "grouped"])
    bench.add_argument("--warmup", type=int, default=5)
    bench.add_argument("--repeats", type=int, default=25)
    bench.add_argument("--threads", type=int, default=1)
    bench.add_argument("--output", type=Path, help="save complete JSON including raw timing samples")
    profile = commands.add_parser("profile", help="export a Chrome trace of a fixed 128-token workload")
    profile.add_argument("--backend", choices=("pytorch", "grouped", "torch-compiled", "triton"), default="grouped")
    profile.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    profile.add_argument("--output", type=Path, default=Path("traces/grouped.json"))
    args = parser.parse_args(argv)
    try:
        import torch

        if args.command == "doctor":
            info = {
                "python": sys.version.split()[0], "platform": platform.platform(),
                "pytorch": torch.__version__, "cuda_build": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "note": "A CPU PyTorch build cannot enumerate GPU capabilities; this is not a hardware inventory.",
            }
            if args.wgpu:
                from .wgpu_backend import list_adapters
                info["wgpu_adapters"] = list_adapters()
            print(json.dumps(info, indent=2))
        elif args.command == "gpu-bench":
            from .wgpu_benchmark import run_wgpu_benchmark
            report = run_wgpu_benchmark(
                num_tokens=args.tokens, num_experts=args.experts, top_k=args.top_k,
                dim=args.dim, hidden=args.hidden, distribution=args.distribution, seed=args.seed,
                warmup=args.warmup, repeats=args.repeats, threads=args.threads, adapter_name=args.adapter,
            )
            adapter = report['environment']['gpu_adapter']
            print(f"RouteKernel | {adapter['device']} | {adapter['backend_type']} | FP32")
            print(f"{args.tokens} tokens | {args.experts} experts | top-{args.top_k} | {args.distribution}")
            print('Path               p50 ms    p95 ms    input tok/s')
            for name, result in report['results'].items():
                print(f"{name:18} {result['p50_ms']:8.3f}  {result['p95_ms']:8.3f}  {result['tokens_per_second']:11,.0f}")
            result = report['results']['wgpu-request']
            print(f"GPU request / CPU: {result['speedup_vs_cpu_pytorch']:.2f}x vs PyTorch; {result['speedup_vs_cpu_grouped']:.2f}x vs grouped")
            print('Request includes CPU routing, transfers, GPU execution and readback; weights resident.')
            print('Resident excludes routing/input transfers/full output readback; includes command encoding and a 4-byte completion read.')
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
                print(f'Saved report: {args.output}')
        elif args.command == "demo":
            from .execution_demo import main as demo
            demo()
        elif args.command == "profile":
            from .benchmark import profile_workload
            profile_workload(args.output, backend=args.backend, device=args.device)
            print(f"Saved profiler trace: {args.output}")
        else:
            from .benchmark import run_benchmark
            report = run_benchmark(
                num_tokens=args.tokens, num_experts=args.experts, top_k=args.top_k,
                dim=args.dim, hidden=args.hidden, distribution=args.distribution, seed=args.seed,
                device=args.device, dtype=args.dtype, backends=tuple(args.backends),
                warmup=args.warmup, repeats=args.repeats, threads=args.threads,
            )
            print(f"RouteKernel | synthetic | {args.device} | {args.dtype}")
            print(f"{args.tokens} tokens | {args.experts} experts | top-{args.top_k} | {args.distribution}")
            print(f"Expert loads: {report['routing']['expert_counts']}")
            print("Backend          p50 ms    p95 ms    input tok/s   vs PyTorch")
            for backend, result in report["results"].items():
                if result["status"] == "error":
                    print(f"{backend}: ERROR: {result['error']}")
                else:
                    speedup = result.get("speedup_vs_pytorch")
                    relative = f"{speedup:.2f}x" if speedup is not None else "n/a"
                    print(f"{backend:16} {result['p50_ms']:8.3f}  {result['p95_ms']:8.3f}  {result['tokens_per_second']:11,.0f}  {relative}")
            print("Scope: supplied routes through merged output; router scoring and validation excluded.")
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                print(f"Saved report: {args.output}")
            if any(result["status"] != "ok" for result in report["results"].values()):
                return 1
    except (ImportError, RuntimeError, ValueError, AssertionError) as exc:
        print(f"routekernel: {exc}", file=sys.stderr)
        return 1
    return 0
