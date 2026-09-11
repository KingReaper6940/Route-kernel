"""Continue Day 1's routing example with hand-checkable expert arithmetic."""

import torch

from .routing import compile_routes
from .runtime import ExpertWeights, forward, reference_forward


def main() -> None:
    tokens = torch.tensor([[1., 2.], [3., 4.], [-1., 2.]])
    ids = torch.tensor([[2, 0], [1, 2], [0, 2]])
    weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]])
    # Identity up projections and scaled down projections keep the arithmetic
    # visible: expert e computes (e + 1) * relu(x).
    experts = ExpertWeights(
        torch.eye(2).repeat(3, 1, 1),
        torch.stack([torch.eye(2) * scale for scale in (1, 2, 3)]),
    )
    routes = compile_routes(ids, weights, num_experts=3)
    print("Input token vectors:\n", tokens)
    print("Grouped token IDs:", routes.token_ids.tolist())
    print("Expert offsets:", routes.expert_offsets.tolist())
    print("\nExpert 0 = relu(x); expert 1 = 2 * relu(x); expert 2 = 3 * relu(x)")
    print("Token 0: 0.7 * [3, 6] + 0.3 * [1, 2] = [2.4, 4.8]")
    expected = torch.tensor([[2.4, 4.8], [7.2, 9.6], [0., 2.8]])
    reference = reference_forward(tokens, ids, weights, experts)
    torch.testing.assert_close(reference, expected)
    for backend in ("pytorch", "grouped"):
        actual = forward(tokens, ids, weights, experts, backend=backend)
        torch.testing.assert_close(actual, expected)
        print(f"\n{backend} output (matches hand calculation):\n", actual)


if __name__ == "__main__":
    main()
