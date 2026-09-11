# Day 1: make the computation understandable and correct

Day 1's target is a tested reference implementation of one simplified MoE layer.
We will build it in small steps. A full language model and custom GPU kernels
come later.

Steps 3 and 4 are now implemented; their code walkthrough is in the
[Day 2 guide](day-2.md). This page preserves the original learning sequence.

## Step 1 — project foundation

Files introduced:

| File | Purpose |
| --- | --- |
| `pyproject.toml` | Names the Python project and defines its packaging configuration. |
| `routekernel/__init__.py` | Makes `routekernel` importable and records its version. |
| `.gitignore` | Keeps local environments and generated files out of Git. |
| `README.md` | Explains the goal, current status, and local setup. |

The local `.venv` directory is an isolated Python environment, not source code.
It stays on this computer and is ignored by Git.

Checkpoint: importing `routekernel` prints version `0.1.0`.

## Step 2 — follow a tiny routing example

Suppose we have three tokens and three experts. Each token selects two experts
(`top_k = 2`). These assignments will be hand-written so we can understand them
without loading a model:

| Token | Selected experts | Corresponding weights |
| --- | --- | --- |
| 0 | 2, 0 | 0.7, 0.3 |
| 1 | 1, 2 | 0.6, 0.4 |
| 2 | 0, 2 | 0.8, 0.2 |

There are three original tokens but six token-expert assignments. Grouping the
assignments by expert produces:

| Expert | Original token IDs | Weights |
| --- | --- | --- |
| 0 | 0, 2 | 0.3, 0.8 |
| 1 | 1 | 0.6 |
| 2 | 0, 1, 2 | 0.7, 0.4, 0.2 |

The grouped token IDs are `[0, 2, 1, 0, 1, 2]`. The expert offsets are
`[0, 2, 3, 6]`: expert `e` owns the half-open slice from `offsets[e]` to
`offsets[e + 1]`. The final offset is the total number of assignments.

We preserve token IDs and weights so we can combine results afterward. For
example, token 0's output is `0.7 * expert_2(x_0) + 0.3 * expert_0(x_0)`, where
`x_0` is its input vector. Grouping changes processing order; it must preserve
this calculation.

### Run and read the example

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m routekernel.routing_demo
```

The implementation is in [`routekernel/routing_demo.py`](../routekernel/routing_demo.py).
Read `main()` in three sections:

1. **Flatten.** `enumerate` gives each token its ID. `zip` pairs each selected
   expert with its weight. Each `Route` holds one complete assignment. The
   token's two choices become two rows; there are still only three input tokens.
2. **Group.** `sorted` orders the rows by `expert_id`. Sorting the whole `Route`
   keeps its token ID and weight attached. This moves metadata; we have not yet
   copied any token vectors or executed an expert.
3. **Compute offsets.** Count the rows for each expert: `[2, 1, 3]`. Start at
   zero and add each count: `0`, `0 + 2`, `2 + 1`, `3 + 3`. The result is
   `[0, 2, 3, 6]`.

Offsets are positions in the grouped list, not token IDs. Expert 1 owns
`grouped_routes[2:3]`: just the row at position 2, which belongs to token 1.
We need four boundaries for three experts so the last group also has an end.
An unused expert would have equal start and end offsets, giving an empty slice.

Why group? Later, we can collect all vectors assigned to an expert into one
matrix and process them together. This example shows how to identify those
groups; it makes no performance claim about this Python sort.

The weights are hand-written router outputs for this example, and each token's
weights sum to one. We preserve them exactly; grouping does not recalculate or
normalize them.

Checkpoint: the demo prints grouped token IDs `[0, 2, 1, 0, 1, 2]`, weights
`[0.3, 0.8, 0.6, 0.7, 0.4, 0.2]`, and offsets `[0, 2, 3, 6]`.

## Step 3 — build the PyTorch reference

Start with a straightforward loop over tokens and their selected experts. Use
small synthetic tensors and a simplified expert network. This readable version
defines the expected outputs for subsequent implementations.

Checkpoint: calculate and explain the weighted outputs for a tiny input.

## Step 4 — implement and verify grouped execution

Introduce a `RoutedBatch` representation, group assignments by expert, execute
each expert's batch, and accumulate weighted results into original token order.

Compare grouped execution against the reference with numerical tolerances.
Cover uneven expert loads, unused experts, and top-k routing.

Checkpoint: both implementations agree on deterministic and randomized inputs.

## After Day 1

Profile the correct implementation, establish measured baselines, and then work
on GPU kernels. Timing and speedup claims require actual measurements, a stated
workload, and a recorded execution environment.
