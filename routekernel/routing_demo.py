"""Trace a fixed top-2 routing example using only Python.

Run from the repository root: python -m routekernel.routing_demo
This is a teaching example; it does not execute expert networks.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    """One token's assignment to one expert, with its output weight."""

    token_id: int
    expert_id: int
    weight: float


def print_routes(routes: list[Route]) -> None:
    print("  token  expert  weight")
    for route in routes:
        print(f"  {route.token_id:5}  {route.expert_id:6}  {route.weight:6.1f}")


def main() -> None:
    num_experts = 3
    expert_ids = [[2, 0], [1, 2], [0, 2]]
    routing_weights = [[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]]

    # 1. Flatten: turn each token's two choices into separate assignments.
    routes = []
    for token_id, token_experts in enumerate(expert_ids):
        for expert_id, weight in zip(
            token_experts, routing_weights[token_id], strict=True
        ):
            routes.append(Route(token_id, expert_id, weight))

    print("1. Flattened routes (in original token order)")
    print_routes(routes)
    print(f"  {len(expert_ids)} tokens x 2 experts = {len(routes)} assignments")

    # 2. Group: sort entire assignments so weights stay with their tokens.
    # Python's stable sort preserves the original order within each expert.
    grouped_routes = sorted(routes, key=lambda route: route.expert_id)

    print("\n2. Routes grouped by expert")
    print_routes(grouped_routes)

    # 3. Count assignments per expert, then accumulate their group boundaries.
    expert_counts = [0] * num_experts
    for route in grouped_routes:
        expert_counts[route.expert_id] += 1

    expert_offsets = [0]
    for count in expert_counts:
        expert_offsets.append(expert_offsets[-1] + count)

    print("\n3. Group boundaries")
    print("  Grouped token IDs:", [route.token_id for route in grouped_routes])
    print("  Grouped weights:  ", [route.weight for route in grouped_routes])
    print("  Expert counts:    ", expert_counts)
    print("  Expert offsets:   ", expert_offsets)

    # A half-open slice includes its start and excludes its end.
    for expert_id in range(num_experts):
        start = expert_offsets[expert_id]
        end = expert_offsets[expert_id + 1]
        token_ids = [route.token_id for route in grouped_routes[start:end]]
        print(f"  Expert {expert_id}: positions [{start}:{end}] -> tokens {token_ids}")

    print("\nNext: run each expert on its tokens and merge weighted outputs.")
    print("  Token 0 output = 0.7 * expert_2(x_0) + 0.3 * expert_0(x_0)")


if __name__ == "__main__":
    main()
