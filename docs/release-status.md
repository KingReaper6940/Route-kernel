# RouteKernel v1.0 — release scope

This release is a shareable experimental GPU runtime with a professional
showcase. It answers a bounded question: can compiling synthetic sparse routes
into grouped workloads improve execution relative to our PyTorch baselines?
It is not a completed production LLM serving system.

## Implemented and evidenced

| Area | Delivered |
| --- | --- |
| Routing | Linear top-k router; stable expert grouping; offsets; token permutation and inverse; weighted output reconstruction |
| Correctness | Hand-checkable reference, randomized runtime checks, empty/odd/unused-expert cases, mandatory remote numerical gate |
| NVIDIA | Two Triton grouped projection kernels; A40 checks in FP32/FP16/BF16; smoke run and eight FP16 sweep cases |
| AMD | Vulkan/WGSL gather, tiled expert projections, and merge; RX 6800 hardware validation in FP32 |
| Baselines | PyTorch, grouped PyTorch, and compilation of expert arithmetic; timings reported without silent fallbacks |
| Evidence | Raw timing samples, p50/p95, environment metadata, source fingerprints, correctness records, and GPU trace |
| Remote workflow | Source packaging, setup, GPU preflight, correctness gate, bounded test execution, downloadable result archives |
| Communication | Architecture walkthroughs, limitations, original CPU and GPU reports, interactive results and routing showcase |
| Deployment | Repository-root Vercel configuration, locked npm dependencies, local production build and browser tests |

The GPU evidence remains the September 12 experiment. Website development and
offline trace analysis do not create a new GPU result. The first rental was
terminated after the results were downloaded.

Local release checks on September 13: 41 Python tests ran, 40 passed and the
NVIDIA-only test skipped on Windows. Compiled CPU and RX 6800 hardware tests
were enabled and passed. Three website data/simulation tests and four browser
tests passed. A clean `npm ci` followed by the production build succeeded.
The NVIDIA evidence is the earlier remote run, not the Windows test skip.

## Findings worth explaining

- A40 Triton median latency was lower than grouped PyTorch in seven of eight
  sweep cases; the observed ratio ranged from 0.97x to 1.63x.
- Compiling only the expert MLP was slower in that sweep. It leaves Python
  dispatch, routing, and merging outside compilation.
- The trace has 29 GPU kernel events, including two custom projection kernels.
  [The profile analysis](profile-analysis.md) explains why this motivates more
  investigation without pretending profiler durations are benchmark latency.
- The AMD request benchmark includes host transfers; the NVIDIA benchmark
  starts with GPU-resident inputs. Their speedups are not interchangeable.

## Further research, outside the measured release claims

1. Repeated GPU runs with multiple seeds and randomized backend order, followed
   by uncertainty estimates. One fixed-order session is insufficient for a
   robust performance claim.
2. Stage-level measurements and routing/dispatch optimization, including
   reducing metadata construction, avoiding host synchronization, and fusing
   operations where measurements justify it. Any new GPU code needs another
   hardware correctness gate before publishing a speedup.
3. A pretrained MoE adapter with the actual model's expert architecture and
   router semantics; replay of real model routing traces; comparison against
   the original model implementation. The current ReLU experts are not an
   OLMoE adapter.
4. End-to-end text generation and a stronger production MoE baseline. Training,
   distributed expert placement, capacity limits, and token dropping are
   separate extensions.

The showcase makes these boundaries visible. Do not use the aspirational
OLMoE, full-model throughput, or placeholder benchmark numbers from the
original project pitch as release claims.
