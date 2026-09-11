# Day 2: execute the assignments correctly

Day 1 stopped after grouping metadata. We first finish its missing execution
and correctness steps, then turn the example into reusable runtime code.

## 1. Replace labels with vectors

A token arriving at an MoE layer is a vector. Our teaching input has three
tokens, each with two numbers:

```text
x = [[ 1, 2],
     [ 3, 4],
     [-1, 2]]
```

Its shape is `[T, D] = [3, 2]`: T token rows and D features per row. The model
would normally supply these vectors; this example uses hand-written values.

The original routes and weights stay unchanged:

```text
ids     = [[2, 0], [1, 2], [0, 2]]
weights = [[.7, .3], [.6, .4], [.8, .2]]
```

## 2. Give each expert a computation

Read `expert_mlp` in [`runtime.py`](../routekernel/runtime.py):

```python
torch.relu(tokens @ up) @ down
```

`@` means matrix multiplication. The first multiplication changes the width
from D to H hidden features. ReLU replaces negative values with zero. The
second multiplication changes H back to D. Each expert has different `up`
and `down` parameter matrices. This simple network lets us study execution;
it does not reproduce the gated expert architecture of a specific LLM.

The demo chooses special matrices so expert 0 computes `relu(x)`, expert 1
computes `2 * relu(x)`, and expert 2 computes `3 * relu(x)`. Token 0 produces:

```text
0.7 * [3, 6] + 0.3 * [1, 2] = [2.4, 4.8]
```

The expected full output is `[[2.4, 4.8], [7.2, 9.6], [0, 2.8]]`.
Run `python -m routekernel demo` to see and verify this calculation.

## 3. Write a correctness reference

`reference_forward` loops over each token and each selected expert. It runs
that expert, multiplies its output by the routing weight, and adds it to the
original token's output row. This is deliberately easy to inspect. We compare
other implementations against it; it is not our performance baseline.

## 4. Compile routing metadata

Read [`routing.py`](../routekernel/routing.py). `compile_routes` validates the
inputs, flattens them, sorts by expert ID, counts loads, and builds cumulative
boundaries. It returns `RoutedBatch`:

| Field | Meaning |
| --- | --- |
| `token_ids` | Original token row for each grouped assignment |
| `expert_ids` | Expert for each grouped assignment, in ascending order |
| `expert_offsets` | E + 1 boundaries defining contiguous expert slices |
| `routing_weights` | Weights in the same order as the grouped assignments |
| `permutation` | For each grouped row, the original flattened route index |
| `inverse_permutation` | For each original flattened route, its grouped position |

In our example, the permutation is `[1, 4, 2, 0, 3, 5]`. Original flattened
route 1 is token 0's assignment to expert 0, so it becomes grouped row 0.
The inverse is `[3, 0, 2, 4, 1, 5]`.

`grouped_values[inverse_permutation]` restores **route order**. It does not
merge routes into token outputs; that still requires weighting and summing.

The public function accepts int32/int64 expert IDs, distinct selected experts
per token, and finite nonnegative weights. It preserves supplied weights without
forcing them to sum to one. Empty batches and unused experts are valid. Tokens
and expert matrices must share dtype and device.

## 5. Gather, execute, and merge

`tokens[routes.token_ids]` gathers vectors into expert order. A token chosen by
two experts appears twice. The grouped backend runs each nonempty expert slice
as a matrix operation.

`merge_outputs` multiplies each row by its routing weight and uses `index_add_`
to sum it into the original token row. Simply assigning rows would overwrite
contributions, which is why we need addition.

For FP16/BF16 outputs, weighted summation happens in FP32 before casting back.
Different matrix shapes can produce slightly different rounding, so correctness
checks use numerical tolerances rather than exact bit equality.

## 6. Keep the baseline useful

The `pytorch` backend also batches tokens for each expert. It finds the rows
with `torch.where` for each expert instead of creating one shared sorted plan.
Comparing against this baseline tells us whether the extra grouping work helps.

The grouped PyTorch backend reads offsets into a Python list. On a GPU that
introduces synchronization. This is a readable implementation with a known
limitation; the experimental Triton path looks up boundaries on the device.

## Check your understanding

- Why do three tokens with top-k = 2 produce six grouped rows?
- Why does expert 1 own slice `[2:3]` even though its ID is 1?
- Why must merging add outputs rather than overwrite them?

Run `python -m unittest discover -s tests -v` for hand calculations, randomized
routes, unused experts, empty batches, invalid inputs, noncontiguous tensors,
low precision, and permutation reconstruction.
