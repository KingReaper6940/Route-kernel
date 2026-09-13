# What the first GPU trace tells us

The recorded A40 trace is for one warmed-up synthetic Triton call: 128 tokens,
eight experts, top-2 routing, D64/H128, FP32. It is separate from the FP16 sweep.

```bash
python remote/analyze_profile.py docs/results/nvidia-a40/sweep/profile.json --output docs/results/nvidia-a40/profile-summary.json
```

The analyzer runs offline and requires only Python's standard library.

The trace contains **328 events**, with **29 GPU kernel events**. Two are the
custom `grouped_projection` kernels; 27 belong to surrounding PyTorch work.
There are also five GPU memcpy events. These counts are saved in the
[machine-readable summary](results/nvidia-a40/profile-summary.json).

The grouped projections perform the first expert matrix multiplication plus
ReLU, then the second matrix multiplication. The surrounding work includes
sorting expert IDs, counting assignments, computing offsets and tile counts,
gathering token vectors, weighting outputs, and adding them back into token
rows. A grouped GEMM reduces the expert loop, but does not remove that work.

The useful next hypothesis is that routing and dispatch overhead could matter
for small inputs. The trace shows enough surrounding operations to justify
testing it. It does not establish how much latency a fusion would save.

Profiler spans overlap and CPU operation spans are nested. The trace also
records instrumentation overhead. Adding all event durations would double-count
work; dividing summed kernel time by the outer CPU span would mix different
measurements. The analyzer therefore reports event counts and per-name duration
sums, without inventing an end-to-end speedup or attributing a percentage of
the benchmark to one operation.

A sound follow-up would measure the individual stages and full path using
repeated trials, then change one stage and rerun the numerical gate and the
same workloads. The measured v0.1 implementation remains unchanged here.
