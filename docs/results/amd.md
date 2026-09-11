# AMD GPU measurements — September 11, 2026

RouteKernel's custom WGSL shaders ran on the **AMD Radeon RX 6800 through
Vulkan** on native Windows. The adapter reports `DiscreteGPU`, AMD vendor ID
4098, and driver description `26.3.1 (AMD proprietary shader compiler)`.
Software adapters are rejected by the implementation.

These measurements use wgpu 0.32.0, PyTorch 2.14.0+cpu, Python 3.12.14, and
Windows 11. PyTorch is the CPU baseline, not the GPU executor. The GPU executes
our gather, tiled matrix multiplication, and weighted merge shaders.
The [environment snapshot](amd-environment.txt) and
[source hashes](amd-source-sha256.json) record the tested implementation.

## Measured median wall latency

Every case uses 8 experts, top-2 routing, FP32, seed 0, 10 warmups, and 30 timed
samples. D is input width; H is hidden width. Each case passes a numerical
comparison against the token-by-token reference before timing.

| Routes | Tokens | D / H | CPU threads | CPU PyTorch ms | CPU grouped ms | GPU request ms | GPU resident ms | Request vs CPU grouped |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 128 | 64 / 128 | 1 | 0.406 | 0.260 | 0.939 | 0.369 | 0.28x |
| Uniform | 512 | 256 / 512 | 1 | 4.985 | 4.942 | 1.480 | 0.788 | 3.34x |
| Uniform | 512 | 512 / 1024 | 1 | 19.090 | 18.660 | 3.250 | 1.881 | 5.74x |
| Skewed | 512 | 512 / 1024 | 1 | 16.863 | 16.987 | 3.169 | 1.815 | 5.36x |
| Uniform | 512 | 512 / 1024 | 8 | 5.331 | 5.009 | 3.570 | 1.877 | 1.40x |

The GPU loses on the tiny case: route preparation, buffer creation, transfers,
and synchronization cost more than the saved arithmetic. Larger matrices
amortize those costs. Against the machine's default eight CPU threads, the
large uniform case was 1.40x faster than CPU grouped execution by median
request latency in this run. This is a GPU-versus-CPU comparison, not a claim
of beating another optimized GPU runtime.

The large eight-thread case had **GPU p95 8.709 ms versus CPU grouped p95
5.254 ms**. Its GPU median improved, but its tail latency was worse. GPU p95
outliers also appeared in other cases. These are single local runs on a desktop
GPU without isolation or confidence intervals; they do not establish a robust
production speedup.

## Timing boundaries

- **CPU paths:** supplied CPU routes to merged CPU output, including sorting
  and gathering where applicable. Expert parameters and inputs are already
  allocated. Validation is excluded.
- **GPU request:** CPU route compilation and tile scheduling, request buffer
  creation, token/metadata upload, GPU gather, both expert projections, GPU
  merge, full output readback, and request cleanup. Expert parameters have
  already been uploaded and remain resident.
- **GPU resident:** reuse the prepared buffers; encode and submit the four GPU
  dispatches, then block on a four-byte output read. This excludes route
  preparation, input uploads, and full output readback. The small completion
  copy/map and CPU dispatch costs are included, so this is not pure kernel time.

All timings exclude router scoring, input generation, validation, reference
checks, shader compilation, and initial expert upload. They are steady-state
single-layer synthetic measurements, not text generation throughput. We do not
compute resident-versus-CPU speedup because their input/output boundaries differ.

## Raw reports

- [Small uniform](amd-uniform-128-64.json)
- [Medium uniform](amd-uniform-512-256.json)
- [Large uniform, one CPU thread](amd-uniform-512-512.json)
- [Large skewed, one CPU thread](amd-skewed-512-512.json)
- [Large uniform, eight CPU threads](amd-uniform-512-512-threads8.json)

Each report contains every sample, median, p95, numerical error, routing counts,
adapter identity, and CPU thread count. The original [CPU-only results](README.md)
remain as a separate historical experiment.

## Reproduce the eight-thread case

```powershell
.\.venv\Scripts\python.exe -m routekernel gpu-bench --adapter 'RX 6800' --tokens 512 --experts 8 --top-k 2 --dim 512 --hidden 1024 --threads 8 --warmup 10 --repeats 30 --output benchmark_results/amd-eight-threads.json
```

For the small and medium cases, change tokens/D/H as shown in the table and
use `--threads 1`. Use `--distribution skewed` for the skewed case.

## Validation and remaining work

The complete local suite passed 31 test methods, including compiled CPU
execution and six hardware GPU tests. The original Triton GPU test remains
skipped. GPU tests cover known outputs, odd dimensions, empty workloads, unused
experts, top-k extremes, noncontiguous tensors, zero/unnormalized weights,
repeated dispatch, resource lifetime, and benchmark timing labels.

FP32 on the RX 6800/Vulkan is the validated path. GPU route compilation,
persistent request-buffer reuse, lower precision, tuning, hardware timestamp
profiling, real model integration, and other adapters are still future work.
The four-byte completion fence is a workaround for the installed wgpu queue
callback issue, and its overhead is deliberately included in measurements.
