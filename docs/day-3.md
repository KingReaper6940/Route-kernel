# Day 3: measure execution and prepare the GPU path

**Update:** the [AMD pivot](amd-pivot.md) now provides a working, tested GPU
path on the local RX 6800. The Triton section below describes the earlier
NVIDIA-specific experiment, which remains separate and unvalidated.

The goal is a reproducible experiment: prove correctness, measure actual work,
and record enough context for someone else to interpret the result.

## Benchmark terminology

| Metric | Meaning |
| --- | --- |
| Latency | Time to process one supplied batch of routes |
| p50 | Median measured latency |
| p95 | Latency at the 95th percentile of collected samples |
| Input tokens/s | Original token count divided by p50 time |
| Assignments/s | Token count multiplied by top-k, divided by p50 time |
| Speedup | Eager PyTorch p50 divided by another backend's p50 |
| Expert load | Number of token assignments sent to that expert |
| Load CV | Population standard deviation of loads divided by their mean |

Speedup below 1 means the compared backend is slower. A high load CV means
uneven work distribution. Min/max load and CV include unused experts.

## Run an experiment

```powershell
.\.venv\Scripts\python.exe -m routekernel bench --tokens 128 --experts 8 --top-k 2 --dim 64 --hidden 128 --distribution uniform --warmup 10 --repeats 50 --output benchmark_results/uniform-128.json
```

Try `--distribution balanced` for nearly equal counts or `skewed` for scores
strongly biased toward the first top-k experts. `uniform` means equal selection
probability, not exactly equal realized counts. `--seed` makes inputs and routes
reproducible. These are synthetic inputs, not traces from a real model.

The benchmark validates inputs once and checks each backend against the
reference before timing. The timed region starts with supplied routes and ends
with merged outputs. It includes grouping, gathering, expert execution,
allocation, and merging where applicable. It excludes router scoring,
validation, workload generation, correctness checks, and initial compilation.
Warmup runs are discarded. CUDA timing synchronizes before and after each
sample so launch time is not mistaken for completed execution.

This is wall latency for one synthetic MoE layer with reused inputs. It is not
full-model generation throughput or isolated GPU kernel time. JSON reports
record raw samples, dimensions, dtype, threads, software, device, and numerical
tolerances. Small differences can be noise; compare multiple workloads and
repeat before making performance claims.

## Profile operations

```powershell
.\.venv\Scripts\python.exe -m routekernel profile --backend grouped --output traces/grouped.json
```

The fixed profile workload has 128 tokens, 8 experts, top-2, D = 64, H = 128,
uniform routes, and FP32. It records one pass after warmup and exports a Chrome
trace. A compatible trace viewer can show sorting, indexing, matrix operations,
and accumulation. CUDA runs also request CUDA activities. Profiler overhead
makes this useful for explaining work rather than replacing latency benchmarks.

## Compare compiled expert arithmetic

```powershell
.\.venv\Scripts\python.exe -m routekernel bench --backends pytorch grouped torch-compiled --output benchmark_results/compiled.json
```

`torch-compiled` uses `torch.compile(..., fullgraph=True, dynamic=True)` on the
expert MLP. Routing, the expert loop, and merging remain eager. It is not
whole-layer compilation. The first execution triggers compilation before timing;
the function is cached in the process. Errors are reported, not hidden by fallback.

On Windows, use a Visual Studio Developer PowerShell configured for x64 C++.
For the installed Visual Studio on this development PC:

```powershell
& 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$env:ROUTEKERNEL_TEST_COMPILED = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

This configures the current shell only. Other machines should use their own
Developer PowerShell or an appropriate C++ compiler.

## Experimental Triton execution

[`triton_backend.py`](../routekernel/triton_backend.py) implements two grouped
matrix multiplications, one per expert projection. GEMM means general matrix
multiplication. A **tile** is a small rectangular block of output computed by
one GPU program.

Each expert has a different number of assigned token rows. We count its row
tiles, compute boundaries, and let GPU programs locate their experts using
those boundaries. Extra programs exit immediately. This avoids reading group
sizes back to Python. Masked loads and stores protect partial tiles, including
odd feature dimensions and short expert groups.

The first projection includes ReLU; the second projects back to D features.
Tile accumulation uses FP32. Routing, vector gathering, and weighted merging
remain PyTorch operations. Tile sizes are fixed without autotuning, so this is
an experimental implementation rather than a tuned kernel.

The backend currently requires Linux, CUDA PyTorch, and NVIDIA hardware. The
local inventory found an AMD Radeon RX 6800; local development uses CPU PyTorch.
The CUDA kernel has therefore not been executed or benchmarked locally. AMD/ROCm
support is separate work; this implementation does not claim it.

On a supported NVIDIA Linux/WSL machine:

1. Install the appropriate CUDA PyTorch build with the
   [official selector](https://pytorch.org/get-started/locally/).
2. Run `python -m pip install -e '.[gpu]'`.
3. Run `python -m routekernel doctor` and confirm CUDA availability.
4. Run `python -m unittest discover -s tests -v`. Confirm the Triton test runs
   rather than skips. It covers short/empty/uneven groups, odd dimensions, FP32,
   FP16, and BF16 when supported.
5. Run `python -m routekernel bench --device cuda --dtype float16 --backends pytorch grouped triton --output benchmark_results/gpu.json`.

Kernel or numerical failures must be resolved before reporting a speedup.
Measure additional realistic shapes and inspect a GPU profiler trace before
making performance claims.

## Before a GPU launch claim

- For the Triton backend specifically, execute and validate it on supported hardware.
- Measure useful GPU shapes and tune from profiling evidence.
- Add model-specific expert architecture and real routes before making OLMoE
  or other pretrained-model claims.

Official references: [PyTorch installation](https://pytorch.org/get-started/locally/),
[Triton installation](https://triton-lang.org/main/getting-started/installation.html),
[grouped GEMM concepts](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html).
