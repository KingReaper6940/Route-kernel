"""Reproducible synthetic workloads; no model downloads or pretrained claims."""

import math

import torch

from .runtime import ExpertWeights


def make_workload(
    num_tokens=128, num_experts=8, top_k=2, dim=64, hidden=128,
    distribution="uniform", seed=0, device="cpu", dtype=torch.float32,
):
    if min(num_experts, dim, hidden) < 1 or num_tokens < 0 or not 1 <= top_k <= num_experts:
        raise ValueError("invalid workload dimensions or top_k")
    if distribution not in ("uniform", "skewed", "balanced"):
        raise ValueError("distribution must be uniform, skewed, or balanced")
    # Generate on CPU so the same seed selects the same routes across devices.
    rng = torch.Generator().manual_seed(seed)
    tokens = torch.randn(num_tokens, dim, generator=rng)
    experts = ExpertWeights(
        torch.randn(num_experts, dim, hidden, generator=rng) / math.sqrt(dim),
        torch.randn(num_experts, hidden, dim, generator=rng) / math.sqrt(hidden),
    )
    scores = torch.randn(num_tokens, num_experts, generator=rng)
    if distribution == "skewed":
        scores[:, :top_k] += 5.0
    if distribution == "balanced":
        ids = torch.arange(num_tokens * top_k).reshape(num_tokens, top_k) % num_experts
        selected = scores.gather(1, ids)
    else:
        selected, ids = scores.topk(top_k, dim=1)
    weights = selected.softmax(dim=1)
    return (
        tokens.to(device=device, dtype=dtype), ids.to(device), weights.to(device),
        ExpertWeights(experts.up.to(device=device, dtype=dtype), experts.down.to(device=device, dtype=dtype)),
    )
