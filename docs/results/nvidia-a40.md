# First NVIDIA experiment — September 12, 2026

RouteKernel's existing Triton backend passed its first real NVIDIA run on a
Runpod **NVIDIA A40**. No kernel changes were needed. In the eight-case FP16
sweep, Triton had a lower median latency than grouped PyTorch in seven cases,
with observed speedups from **0.97x to 1.63x**. The small skewed case regressed
by about 3%. These are synthetic single-layer measurements on one rented GPU.

## Hardware and verification

The rental used Secure Cloud in CA-MTL-1 at a reported compute rate of $0.49/hour.
RTX 4090 creation failed for lack of capacity, so these are explicitly A40 results.
The image was `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
The recorded environment was Linux, Python 3.12.3, PyTorch 2.8.0+cu128,
Triton 3.4.0, CUDA build 12.8, NVIDIA driver 580.159.04, and GPU capability 8.6.
The image tag is recorded; its registry digest was not independently resolved.

Both the smoke run and sweep passed the environment check, routing tests, CPU
runtime tests, and mandatory numerical gate. Each gate checked 96 combinations:
four backends, three dtypes (FP32, FP16, BF16), four shapes, and two routing
distributions. Of those, 24 explicitly exercised Triton. Cases included empty
inputs, one token, odd dimensions, unused experts, and top-k equal to the number
of experts. The numerical comparisons passed within the recorded dtype-specific
tolerances; this is not a claim of bitwise equality.

- [Smoke correctness](nvidia-a40/smoke/correctness.json) and [run status](nvidia-a40/smoke/status.json).
- [Sweep correctness](nvidia-a40/sweep/correctness.json) and [run status](nvidia-a40/sweep/status.json).
- [Environment](nvidia-a40/sweep/environment.json) and [source fingerprints](nvidia-a40/sweep/source-sha256.json).

The downloaded smoke archive's source fingerprints matched all 32 local source
files before these documentation updates. Each full ZIP also contains the
exact source archive and per-step logs.

## Measurements

All cases used eight experts, top-2 routing, seed 0, one PyTorch CPU thread,
10 warmup calls, and 30 measured calls. The expert is a bias-free
`Linear → ReLU → Linear` network. Each backend received the same inputs and
expert weights on the same A40. CUDA was synchronized before and after each
wall-clock sample; initial compilation was excluded.

The smoke run produced these median latencies:

| Case | PyTorch ms | Grouped ms | Compiled MLP ms | Triton ms | Grouped / Triton |
| --- | ---: | ---: | ---: | ---: | ---: |
| [128 tokens, D64, H128, FP32](nvidia-a40/smoke/small-fp32.json) | 1.1905 | 0.7749 | 1.5045 | 0.5638 | 1.37x |
| [512 tokens, D256, H512, FP16](nvidia-a40/smoke/medium-fp16.json) | 1.4752 | 0.9164 | 1.7744 | 0.7517 | 1.22x |

The sweep used FP16 and hidden size H = 2D. The speedup column compares median
latencies against grouped PyTorch, which was the fastest PyTorch variant in
every sweep case. A value below 1 means Triton was slower.

| Tokens | D | Routes / raw data | Grouped p50 ms | Triton p50 ms | Grouped / Triton | Grouped p95 ms | Triton p95 ms |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 256 | [Uniform](nvidia-a40/sweep/t128-d256-uniform.json) | 0.8923 | 0.7533 | 1.18x | 0.9172 | 0.8363 |
| 128 | 256 | [Skewed](nvidia-a40/sweep/t128-d256-skewed.json) | 0.5621 | 0.5801 | 0.97x | 0.5816 | 0.5908 |
| 512 | 256 | [Uniform](nvidia-a40/sweep/t512-d256-uniform.json) | 0.9610 | 0.5908 | 1.63x | 0.9689 | 0.6008 |
| 512 | 256 | [Skewed](nvidia-a40/sweep/t512-d256-skewed.json) | 0.6124 | 0.5679 | 1.08x | 0.6445 | 0.5858 |
| 128 | 512 | [Uniform](nvidia-a40/sweep/t128-d512-uniform.json) | 0.9611 | 0.8992 | 1.07x | 0.9776 | 0.9259 |
| 128 | 512 | [Skewed](nvidia-a40/sweep/t128-d512-skewed.json) | 0.6297 | 0.5943 | 1.06x | 0.6356 | 0.6073 |
| 512 | 512 | [Uniform](nvidia-a40/sweep/t512-d512-uniform.json) | 0.8770 | 0.5872 | 1.49x | 0.9123 | 0.6879 |
| 512 | 512 | [Skewed](nvidia-a40/sweep/t512-d512-skewed.json) | 0.6065 | 0.5585 | 1.09x | 0.6211 | 0.5756 |

Against the per-expert-mask PyTorch baseline, Triton's observed sweep speedup
was 1.24x–2.67x. Compiling only the expert MLP was slower than both eager
PyTorch variants in this sweep. The [full sweep summary](nvidia-a40/sweep/SUMMARY.md)
contains all four backends; the JSON files preserve every timing sample.

## What this establishes, and its limits

The experiment establishes that the Triton implementation executes correctly
on the tested A40 configurations and can reduce latency relative to this
repository's PyTorch baselines. It does not establish performance against a
production MoE library or a fully compiled/fused MoE layer.

The measured scope starts with supplied routes and ends with merged outputs.
It includes grouping, gathering, expert arithmetic, merging, allocations, and
Python overhead as applicable. It excludes router scoring, input validation,
input generation, initial compilation, host-to-device input transfer, and
device-to-host output transfer. Expert parameters and input tensors are already
on the GPU and reused. These numbers cannot be compared directly with the AMD
request benchmark, which includes uploads and full output readback.

Each workload ran in a fresh process, but its backends were measured in fixed
order. This was one rental session with one random seed, no randomized backend
order, and no confidence intervals. The repeated 512/D256 uniform case had
Triton medians of 0.7517 ms in smoke and 0.5908 ms in the sweep, illustrating
run-to-run variation. Small differences should not be treated as reliable wins.
There are no full-model, token-generation, training, or RTX 4090 results here.

## Profile and next experiment

The [Chrome trace](nvidia-a40/sweep/profile.json) contains 328 events, including
29 GPU kernel events. Two are the custom `grouped_projection` launches; the
remaining 27 belong to the surrounding PyTorch execution. This profile uses
the profiler command's fixed 128-token, D64/H128, FP32 workload, separately
from the FP16 sweep. Profiler overhead makes its durations unsuitable as the
benchmark latencies.

The next focused experiment is to investigate routing and dispatch overhead,
then compare a change against these baselines with repeated, randomized-order
measurements. The trace motivates that investigation; it does not by itself
prove the dominant steady-state bottleneck.

## Reproduce and retrieve

Use the [remote harness guide](../remote-nvidia.md), then run in the uploaded
source directory on a configured A40:

```bash
.venv-remote/bin/python remote/run.py --suite smoke --expect-gpu A40 --image-label runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 --output remote-results/a40-smoke
.venv-remote/bin/python remote/run.py --suite sweep --expect-gpu A40 --image-label runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 --output remote-results/a40-sweep --profile
```

Full ZIPs are stored locally at `remote-results/runpod-20260912/a40-smoke.zip`
and `remote-results/runpod-20260912/a40-sweep.zip`. Their
[SHA-256 checksums](nvidia-a40/archives-sha256.json) are included with this report.
Raw JSON, logs, and the trace are preserved alongside the report for review.

After verifying the downloaded archives, the test Pod was terminated. At
23:56:19 UTC Runpod returned empty Pod and network-volume lists. Billing had
not posted a record at the subsequent check, so the final charge is unconfirmed.
