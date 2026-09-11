# Recorded CPU experiment — September 11, 2026

This is the historical CPU-only snapshot. The subsequent AMD pivot has
[separate GPU results](amd.md); the original measurements below are unchanged.

These are local, synthetic single-layer CPU measurements. They are not GPU
results, full-model generation rates, or measurements of OLMoE.

All workloads use 8 experts, top-2 routing, 64 input features, 128 hidden
features, FP32, and seed 0. Each backend receives the same inputs and weights.
The machine ran Windows 11, Python 3.12.14, PyTorch 2.14.0+cpu, and one PyTorch
CPU thread. The processor reports `AMD64 Family 25 Model 97 Stepping 2,
AuthenticAMD`. Compiled execution used the installed Visual Studio x64 C++
toolchain. Package versions are in [environment.txt](environment.txt).

Each case used 10 warmup calls followed by 50 measured calls, after passing
the numerical reference check. The table shows median latency in milliseconds.
JSON files preserve all samples, p95, errors, throughput, routing loads, and
environment details. The recorded source fingerprints are in
[source-sha256.json](source-sha256.json).

| Routes | Tokens | PyTorch ms | Grouped ms | Compiled MLP ms | Grouped speedup | Raw report |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Uniform | 32 | 0.269 | 0.178 | 0.552 | 1.51x | [JSON](cpu-uniform-32.json) |
| Uniform | 128 | 0.343 | 0.249 | 0.464 | 1.38x | [JSON](cpu-uniform-128.json) |
| Uniform | 512 | 0.656 | 0.548 | 0.804 | 1.20x | [JSON](cpu-uniform-512.json) |
| Skewed | 32 | 0.117 | 0.120 | 0.199 | 0.98x | [JSON](cpu-skewed-32.json) |
| Skewed | 128 | 0.194 | 0.193 | 0.279 | 1.00x | [JSON](cpu-skewed-128.json) |
| Skewed | 512 | 0.511 | 0.505 | 0.663 | 1.01x | [JSON](cpu-skewed-512.json) |

The grouped implementation helped on these uniform workloads. Skewed cases
were effectively tied, including one small regression. Both eager backends
skip unused experts. That matters: avoiding empty work only in the grouped
backend would exaggerate its improvement for skewed routing.

Compiling the expert MLP was slower than eager PyTorch in these six cases.
The experiment compiles expert arithmetic only; Python dispatch, routing, and
merging remain eager. These timings do not establish which operation caused
the difference or predict the outcome for larger shapes or GPU execution.

## Measurement limits

- Timing starts with supplied routes and ends with merged outputs. It includes
  grouping, gather, expert execution, and merge as applicable. Router scoring,
  input validation, generation, correctness checks, and initial compilation
  are excluded.
- This is a steady-state microbenchmark reusing the same tensors. It does not
  measure model loading, attention, tokenization, KV caches, or text generation.
- Backends run sequentially in the order shown. Measurements are one local
  snapshot without confidence intervals or process isolation. Small differences
  are not evidence of a reliable improvement.
- The CPU grouped path reads offsets into Python. A GPU implementation has
  different launch, memory-transfer, and synchronization costs.
- No GPU speedup is reported. The experimental Triton kernel has not run on
  this machine. Local hardware includes an AMD RX 6800; this backend currently
  targets CUDA/NVIDIA on Linux.

## Reproduce

Use the setup in [Day 3](../day-3.md) to enable the C++ compiler, then run from
the repository root:

```powershell
foreach ($distribution in @('uniform', 'skewed')) {
    foreach ($tokens in @(32, 128, 512)) {
        .\.venv\Scripts\python.exe -m routekernel bench --tokens $tokens --experts 8 --top-k 2 --dim 64 --hidden 128 --distribution $distribution --backends pytorch grouped torch-compiled --warmup 10 --repeats 50 --output "benchmark_results/cpu-$distribution-$tokens.json"
    }
}
```

## Verification status

The local suite ran 24 test methods: 23 passed, including the optional compiled
test, and the Triton GPU test skipped because its required environment was
unavailable. The randomized runtime test also checks 72 parameter combinations
across two eager backends. The profile command produced a valid Chrome trace
with 249 events for the grouped CPU workload. Package installation, dependency
consistency, the CLI, and Python syntax were checked locally.

A Windows/Linux GitHub Actions workflow is included but has not been run on
GitHub from this session. These were the checks at the time of this CPU-only
snapshot. The AMD GPU path was subsequently validated; Triton validation and
pretrained-model integration remain outstanding.
