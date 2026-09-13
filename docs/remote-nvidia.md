# First NVIDIA rental: a reproducible RouteKernel experiment

The harness completed its first NVIDIA smoke run and eight-case sweep on an
A40 on September 12, 2026. See the [recorded A40 results](results/nvidia-a40.md).
The commands below use the initially recommended RTX 4090; set `--expect-gpu`
to the actual rental's name (for example, `A40`). Its CPU rehearsal verifies
orchestration, logging, failure handling, and packaging separately from GPU
correctness.

## What we will rent

For the initial experiment, use one RTX 4090 (24 GB) on a Linux GPU instance.
Choose a development template with CUDA-enabled PyTorch >= 2.5, its matching
Triton >= 3.1, Python >= 3.10, a C++ compiler, pip, and venv support. Start with
the provider's maintained PyTorch development template and record the actual
image tag/digest in the run. The runner checks capability >= 8.0, GPU identity
when requested, CUDA arithmetic, and required imports.

The setup never installs or changes system GPU drivers. This harness does not
rent, provision, authenticate to, stop, or terminate any cloud resource.

## 1. Upload the current source

The repository's current implementation may include uncommitted files. The
source bundle packages those files directly rather than relying on a GitHub
checkout. From this Windows project directory:

```powershell
.\.venv\Scripts\python.exe remote/bundle.py
```

This creates `dist/routekernel-remote.tar.gz` and its `.sha256` file. It includes
runtime code, WGSL assets, tests, remote scripts, packaging metadata, and this
guide. It excludes `.git`, local environments, `.env`, caches, and previous
experiment outputs. It does not scan source code for embedded credentials;
only put shareable project code in the included source directories.

The default bundle refuses to overwrite an existing archive. After editing,
create a new name with `--output dist/routekernel-remote-v2.tar.gz`.

Once a rental exists, upload the archive and checksum through its file browser
or SSH/SCP. The provider will supply the actual host and port; we have not
invented or saved credentials. In its Linux terminal:

```bash
cd /workspace
sha256sum -c routekernel-remote.tar.gz.sha256
tar -xzf routekernel-remote.tar.gz
cd routekernel-remote
```

Use a fresh extraction directory for a new code bundle. The archive has one
top-level directory, `routekernel-remote`, and a per-file source manifest.

## 2. Configure the Python environment

On a prepared CUDA/PyTorch development template:

```bash
python3 remote/setup.py
```

The default **reuse** mode checks the template first, creates `.venv-remote`
with access to its installed CUDA PyTorch/Triton, installs lightweight packaging
requirements and the local project, and checks the resulting environment.
It deliberately does not replace a template's working PyTorch/Triton pair.

If the template has no suitable Python stack, fresh mode is available:

```bash
python3 remote/setup.py --mode fresh --torch-version 2.13.0 --cuda-wheel cu126
```

This creates an isolated environment and installs the selected official CUDA
PyTorch wheel, including its matching Triton dependency, followed by the
project. The chosen driver must support that wheel; the final CUDA check must
pass. Fresh downloads can take time. Python 3.12 is a reasonable starting point
for the fresh recipe. This fresh installation has not been executed locally.

`--dry-run` prints either recipe without installing anything. Existing
`.venv-remote` directories are preserved: reuse a successful environment, or
use a fresh extraction for a new setup. Setup recipes and logs are saved under
`remote-results/setup-*`. Installation time is outside the runner's deadline.

## 3. Run the small smoke experiment first

```bash
.venv-remote/bin/python remote/run.py --suite smoke --expect-gpu 'RTX 4090' --image-label 'REPLACE_WITH_ACTUAL_TEMPLATE_TAG_OR_DIGEST'
```

The image label is supplied by you and recorded as unverified metadata. It is
not a claim that the script independently obtained the provider's image digest.

The runner executes these stages sequentially:

1. **Preflight:** verify Linux, CUDA, Triton, the compiler, GPU capability, and
   a tiny CUDA matrix multiplication. Record package versions, GPU/driver
   details, available memory, temperature, and utilization when available.
2. **CPU correctness tests:** verify assignment grouping, reconstruction,
   hand-calculated expert outputs, and randomized runtime cases.
3. **Numerical gate:** explicitly execute PyTorch, grouped PyTorch, compiled
   expert arithmetic, and Triton on the GPU against a CPU reference. Test
   empty inputs, a single expert, odd dimensions, unused experts, and top-k
   extremes in FP32/FP16, plus BF16 when the device supports it. There is no
   skipped Triton test being treated as a passing GPU check.
4. **Benchmarks:** run the two workloads in the smoke preset. Every backend
   runs on the same GPU and must pass the existing correctness gate again.
5. **Archive:** save results and source in a downloadable ZIP, including when
   a step fails or times out.

A failure stops subsequent stages. Read `SUMMARY.md`, the failed step's log,
and `status.json` before retrying. No eager fallback is substituted for a failed
compiled or Triton backend. Diagnostic settings such as `TRITON_INTERPRET=1`
and `CUDA_LAUNCH_BLOCKING=1` are rejected for measured runs.

## 4. Run the sweep only after the smoke succeeds

```bash
.venv-remote/bin/python remote/run.py --suite sweep --expect-gpu 'RTX 4090' --profile --max-seconds 1800 --step-timeout 600
```

The sweep has eight FP16 cases across 128/512 tokens, 256/512 input features,
twice as many hidden features, and uniform/skewed routes. All use eight experts
and top-2 routing. Presets live in `remote/suites.json`. They are synthetic
single-layer workloads, not model traces. `--profile` additionally records a
fixed 128-token FP32 Triton workload through PyTorch's profiler.

The default is 10 warmup calls and 30 measured calls. Use `--warmup`,
`--repeats`, and `--seed` to change these explicitly. Each benchmark uses a
fresh process; any initial compilation is excluded from timed samples.
All backend comparisons use the same GPU, dtype, and workload within a case.

The measured scope is supplied routes through merged outputs, including
grouping, gathering, expert execution, and merging as applicable. Router
scoring, input generation, validation, and initial compilation are excluded.
The compiled backend covers expert arithmetic only. CUDA synchronization makes
the wall timer measure completed execution, not just queued launches.

## 5. Download results and terminate the rental

The runner prints the result directory and its adjacent ZIP path. Download the
ZIP before terminating the instance. It contains:

- `SUMMARY.md`, `status.json`, and per-stage logs.
- `environment.json`: actual Python/PyTorch/Triton versions and GPU metadata.
- `correctness.json` and benchmark JSON files with all raw timing samples.
- `source-sha256.json` and `source.tar.gz`: the exact included source used by
  the run, including remote edits.
- `profile.json` if profiling was requested and succeeded.

Each invocation creates a unique directory unless `--output` names a new path.
Existing runs are never overwritten. The runner records installed versions and
source snapshots; an arbitrary mutable template is not a fully pinned runtime.
Preserve the template digest and environment report for a subsequent replay.

**The deadline limits child step execution, not the cloud bill.** On Linux a
timeout kills the step's process group, including compiler workers. Setup,
packaging, and an idle instance are outside that execution deadline. The script
does not stop or terminate the rental. Use the provider console to terminate it
when finished; storage may have a separate lifecycle and charge.

## Local rehearsal without a rental

```powershell
.\.venv\Scripts\python.exe remote/run.py --dry-run --suite smoke
.\.venv\Scripts\python.exe remote/run.py --mode cpu-smoke --warmup 1 --repeats 3
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_remote.py -v
```

The CPU rehearsal produces an archive labelled **NOT GPU VALIDATION**. A normal
GPU-mode invocation on this Windows/CPU-PyTorch environment instead fails
preflight and saves diagnostics. Neither command contacts a GPU provider.

Sources: [Runpod templates](https://docs.runpod.io/pods/templates/overview),
[official PyTorch wheel recipes](https://pytorch.org/get-started/previous-versions/),
and [Triton compatibility](https://github.com/triton-lang/triton#compatibility).
