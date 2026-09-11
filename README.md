# RouteKernel

An experimental runtime for sparse Mixture-of-Experts (MoE) inference.

The engineering question: can we turn sparse token routing into faster GPU
execution by grouping work by expert?

## Intended execution pipeline

Input token vectors → top-k expert assignments → group tokens by expert →
execute expert networks → merge weighted outputs.

An expert is a small neural network. A router selects a few experts for each
token and assigns weights to their outputs. Grouping those assignments gives
each expert a batch of token vectors to process together.

## Current status

**AMD pivot:** RouteKernel now executes custom GPU shaders on the local
**Radeon RX 6800 through Vulkan**, using native `wgpu`. GPU gather, both expert
projections, and weighted merging have passed hardware correctness tests.
Routing metadata is still compiled on the CPU. See the
[AMD walkthrough](docs/amd-pivot.md) and [GPU results](docs/results/amd.md).

Days 2–3 add a linear top-k router, reusable route compiler, expert execution,
weighted merging, correctness tests, benchmarks, and profiling. The experts use
a bias-free `Linear → ReLU → Linear` network on synthetic data.

| Backend | Execution | Status |
| --- | --- | --- |
| `pytorch` | Find and batch each expert's token rows | Tested CPU baseline |
| `grouped` | Sort routes, gather once, execute contiguous expert slices, merge | Tested on CPU |
| `torch-compiled` | Grouped plan with compiled expert arithmetic | Requires a configured C++ compiler on CPU |
| `WgpuEngine` | CPU route planning; custom WGSL gather, tiled expert GEMMs, and merge on GPU | Tested on AMD RX 6800 / Vulkan, FP32 |
| `triton` | Two custom grouped GEMM launches with device-side expert lookup | Experimental; GPU execution unvalidated |

The token-by-token reference is used for correctness, not speedup claims.
Unsupported backends report an error, exit nonzero, and never silently fall back.

This is a single-layer synthetic experiment, not a full LLM runtime or an OLMoE
adapter. PyTorch remains the CPU reference; the wgpu engine supplies the locally
verified GPU path. See the [original CPU results](docs/results/README.md) and
[new AMD GPU results](docs/results/amd.md) for measurements and limitations.

## Local setup

For the AMD GPU path, add the optional dependency to the existing environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[amd]'
.\.venv\Scripts\python.exe -m routekernel doctor --wgpu
.\.venv\Scripts\python.exe -m routekernel gpu-bench --adapter 'RX 6800' --tokens 512 --dim 256 --hidden 512 --output benchmark_results/amd.json
```

This uses native GPU compute on Windows, with no browser or CUDA requirement.
FP32 CPU tensors are uploaded to GPU buffers, and the final output is returned
as a CPU tensor. Expert parameters remain resident during the GPU benchmark.

Run commands from this directory (the one containing `pyproject.toml`).

If `.venv` does not exist, create it with an installed Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e .
```

Check the package with the environment's Python. Activation is optional:

```powershell
.\.venv\Scripts\python.exe -c "import routekernel; print(routekernel.__version__)"
```

Expected output: `0.1.0`.

The existing workspace already has `.venv` and an editable install. Source edits
are immediately available to the installed command. On Linux, use
`.venv/bin/python` instead of `.venv\Scripts\python.exe`.

Run the routing example:

```powershell
.\.venv\Scripts\python.exe -m routekernel.routing_demo
```

Run expert computation, correctness tests, benchmarks, and profiling:

```powershell
.\.venv\Scripts\python.exe -m routekernel demo
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m routekernel doctor
.\.venv\Scripts\python.exe -m routekernel bench --tokens 128 --experts 8 --top-k 2 --output benchmark_results/cpu.json
.\.venv\Scripts\python.exe -m routekernel profile --output traces/grouped.json
```

After activating the environment, `routekernel bench ...` is equivalent to
`python -m routekernel bench ...`.

Learn the implementation in order:

- [Day 1](docs/day-1.md): token assignments and offsets.
- [Day 2](docs/day-2.md): tensors, the reference, grouping, and merging.
- [Day 3](docs/day-3.md): benchmarks, profiling, compilation, and GPU setup.
- [AMD pivot](docs/amd-pivot.md): custom shader execution on the RX 6800.

The package is inference-only. Training, capacity limits, token dropping,
distributed expert placement, and integration with pretrained models remain
future work. The older Triton experiment still targets Linux with CUDA/NVIDIA;
the active local GPU path is `WgpuEngine`. Other hardware and FP16/BF16 shader
paths are not yet validated.
