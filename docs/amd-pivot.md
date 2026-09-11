# RouteKernel on AMD: native GPU compute with wgpu

The Radeon RX 6800 can run our custom kernels. The original implementation's
CUDA requirement was a software choice, not a statement that AMD cannot do GPU
computing. We now have a tested native Windows path using Vulkan through wgpu.
It does not require a browser, changing drivers, or reinstalling PyTorch.

## The vocabulary

| Term | Meaning in this project |
| --- | --- |
| Vulkan | The graphics/compute API communicating with the AMD driver |
| WebGPU | The portable GPU API model implemented by wgpu |
| wgpu-py | Python bindings to the native wgpu implementation |
| WGSL | The language used for our custom GPU shader programs |
| Compute shader / kernel | A program executed in parallel on GPU data |
| Workgroup | Invocations that cooperate and share fast local memory |
| Tile | A small matrix block computed by one workgroup |
| GPU buffer | GPU-accessible storage holding vectors, parameters, or metadata |
| Resident data | Data already uploaded to the GPU and reused across execution |

The name WebGPU does not mean our program runs in a web page. Here Python calls
a native library, which selects the RX 6800's Vulkan adapter. The adapter's
reported device name, backend, driver, and hardware type are saved in results.
CPU/software adapters are rejected. An explicit adapter name that cannot be
found produces an error.

## What changed

The route representation, expert math, and reference tests stay the same.
The execution path is now:

```text
CPU: validate routes → group assignments → schedule expert tiles
                       ↓ upload token vectors and routing metadata
GPU: gather vectors → grouped GEMM + ReLU → grouped GEMM → weighted merge
                       ↓ read back output
CPU: final token vectors
```

An engine uploads expert weights once and reuses them for subsequent requests.
`prepare` creates a request's metadata and buffers. `run` dispatches GPU work
and returns a CPU tensor. `submit` can leave outputs on the GPU for resident
measurements. Parameters are a snapshot: recreate the engine after changing
expert weights. Keep the engine open while using its prepared workloads.

## Read the code in this order

1. [`wgpu_backend.py`](../routekernel/wgpu_backend.py): hardware selection,
   engine lifetime, route preparation, uploads, and four dispatches.
2. [`gather.wgsl`](../routekernel/shaders/gather.wgsl): copy token vectors into
   expert order, repeating vectors for multiple selected experts.
3. [`grouped_gemm.wgsl`](../routekernel/shaders/grouped_gemm.wgsl): a 16×16 tiled
   matrix multiplication. Each workgroup belongs to one expert. Shared tiles
   reuse input and parameter loads; barriers keep cooperating invocations in
   step. Out-of-range elements load zeros, and invalid output positions are
   not written. The same shader runs twice, with ReLU enabled only for the
   first projection.
4. [`merge.wgsl`](../routekernel/shaders/merge.wgsl): use the inverse permutation
   to find each token's expert results, multiply by routing weights, and sum.
   One invocation owns each output element, so no float atomics are needed.
5. [`wgpu_benchmark.py`](../routekernel/wgpu_benchmark.py): correctness gates
   and measurements with explicit transfer boundaries.

For the tiled shader, the CPU route compiler creates `tile_experts` and
`tile_rows`. These tell each workgroup which expert and starting row it owns.
That avoids spending GPU time on padded rows for unused experts.

## Run it

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[amd]'
.\.venv\Scripts\python.exe -m routekernel doctor --wgpu
$env:ROUTEKERNEL_TEST_WGPU = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_wgpu.py -v
.\.venv\Scripts\python.exe -m routekernel gpu-bench --adapter 'RX 6800' --tokens 512 --dim 256 --hidden 512 --output benchmark_results/amd.json
```

The environment in this workspace is already installed. The GPU test flag
enables real hardware tests; it does not silently skip a broken adapter or
missing GPU package. Without the flag, ordinary CPU CI skips hardware tests.
The explicit adapter is optional; by default selection prefers a discrete
hardware GPU and then Vulkan.

Python usage:

```python
from routekernel.workloads import make_workload
from routekernel.wgpu_backend import WgpuEngine

x, expert_ids, weights, experts = make_workload(128, 8, 2, 64, 128)
with WgpuEngine(experts, adapter_name="RX 6800") as engine:
    output = engine.forward(x, expert_ids, weights)
```

## Understand the two GPU timings

`wgpu-request` starts from CPU token vectors and supplied routing decisions. It
includes CPU route compilation, tile scheduling, buffer setup, token/metadata
upload, all four GPU dispatches, full output readback, and request cleanup.
Initial shader compilation and expert-weight upload are excluded.

`wgpu-resident` reuses prepared buffers. It includes command encoding, the four
dispatches, and a blocking four-byte output read to confirm completion. It
excludes route compilation, input uploads, and full output readback. It is
**not isolated kernel time**, and we do not compare its latency directly with
the CPU path to claim full-request speedup.

The four-byte completion read avoids a queue-completion callback ABI error
observed with wgpu 0.32.0 on this environment. Its staging/map overhead is part
of every resident sample. A submission without a completion check would only
measure how quickly Python queues work.

Both CPU baselines return CPU outputs. The request speedups name their CPU
baseline explicitly. `--threads` controls PyTorch CPU threads; the report
records this. The new results include both one-thread and eight-thread cases.

## What is verified, and what remains

Verified locally: hardware adapter selection; the hand-calculated example;
empty inputs; unused experts; odd dimensions; top-k of 1 and all experts;
noncontiguous inputs; zero and unnormalized weights; repeated resident runs;
and correctness-gated GPU benchmarks. See [actual measurements](results/amd.md).

The active shader path supports FP32 inference with CPU input/output tensors.
It still uses a CPU route compiler. Shader tiling is an initial implementation,
not a claim of optimal AMD performance. A real model adapter, GPU route
compilation, persistent request buffers, lower precision, and deeper profiling
remain useful next steps. The separate Triton experiment is still unvalidated.

The broad API portability comes from wgpu, but this project currently claims
hardware validation only for the RX 6800/Vulkan combination tested here.

References: [wgpu's official guide](https://wgpu-py.readthedocs.io/en/stable/guide.html)
and [compute utilities](https://wgpu-py.readthedocs.io/en/stable/utils.html).
