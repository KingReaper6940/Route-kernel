"""Turn token-major top-k assignments into contiguous expert groups."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RoutedBatch:
    """Metadata in expert order; permutation maps grouped rows to flat routes.

    Flat route i belongs to token i // top_k. inverse_permutation restores
    flat route order: grouped_values[inverse_permutation]. Offsets have E + 1
    entries, so unused experts are represented by equal adjacent offsets.
    """

    token_ids: Tensor
    expert_ids: Tensor
    expert_offsets: Tensor
    routing_weights: Tensor
    permutation: Tensor
    inverse_permutation: Tensor
    num_tokens: int
    top_k: int


def validate_routes(expert_ids: Tensor, routing_weights: Tensor, num_experts: int) -> None:
    if isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts < 1:
        raise ValueError("num_experts must be a positive integer")
    if expert_ids.ndim != 2 or expert_ids.shape != routing_weights.shape:
        raise ValueError("expert_ids and routing_weights must have the same [tokens, top_k] shape")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("expert_ids must have int32 or int64 dtype")
    if not routing_weights.is_floating_point():
        raise ValueError("routing_weights must have floating-point dtype")
    if expert_ids.device != routing_weights.device:
        raise ValueError("expert_ids and routing_weights must be on the same device")
    if not 1 <= expert_ids.shape[1] <= num_experts:
        raise ValueError("top_k must be between 1 and num_experts")
    if torch.any((expert_ids < 0) | (expert_ids >= num_experts)).item():
        raise ValueError("expert ID is outside [0, num_experts)")
    if not torch.isfinite(routing_weights).all().item() or torch.any(routing_weights < 0).item():
        raise ValueError("routing_weights must be finite and nonnegative")
    sorted_ids = expert_ids.sort(dim=1).values
    if torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]).item():
        raise ValueError("each token must select distinct experts")


def _compile_routes(expert_ids: Tensor, routing_weights: Tensor, num_experts: int) -> RoutedBatch:
    """Internal implementation for already validated tensors."""
    num_tokens, top_k = expert_ids.shape
    flat_experts = expert_ids.reshape(-1).long()
    permutation = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[permutation]
    counts = torch.bincount(sorted_experts, minlength=num_experts)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)
    return RoutedBatch(
        token_ids=permutation // top_k,
        expert_ids=sorted_experts,
        expert_offsets=offsets,
        routing_weights=routing_weights.reshape(-1)[permutation],
        permutation=permutation,
        inverse_permutation=inverse,
        num_tokens=num_tokens,
        top_k=top_k,
    )


def compile_routes(expert_ids: Tensor, routing_weights: Tensor, num_experts: int) -> RoutedBatch:
    """Validate and group routes, preserving weights without renormalizing."""
    validate_routes(expert_ids, routing_weights, num_experts)
    return _compile_routes(expert_ids, routing_weights, num_experts)


def topk_route(tokens: Tensor, gate: Tensor, top_k: int) -> tuple[Tensor, Tensor]:
    """A small linear router; softmax normalizes only the selected scores.

    tokens: [T, D], gate: [D, E]. This is a synthetic router, not a pretrained
    model adapter. It returns IDs and weights with shape [T, top_k].
    """
    if tokens.ndim != 2 or gate.ndim != 2 or tokens.shape[1] != gate.shape[0]:
        raise ValueError("tokens [T, D] and gate [D, E] must have matching feature dimensions")
    if not tokens.is_floating_point() or tokens.dtype != gate.dtype or tokens.device != gate.device:
        raise ValueError("tokens and gate must share a floating-point dtype and device")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= gate.shape[1]:
        raise ValueError("top_k must be between 1 and the number of experts")
    scores = tokens @ gate
    values, ids = scores.topk(top_k, dim=-1)
    # Normalize in FP32 for low-precision inputs, retaining FP64 when requested.
    weights = torch.softmax(values.to(accumulation_dtype(tokens.dtype)), dim=-1)
    return ids, weights


def accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def merge_outputs(expert_outputs: Tensor, routes: RoutedBatch) -> Tensor:
    """Weight grouped outputs and sum into original token rows."""
    dtype = accumulation_dtype(expert_outputs.dtype)
    weighted = expert_outputs.to(dtype) * routes.routing_weights.to(dtype).unsqueeze(1)
    output = torch.zeros(
        (routes.num_tokens, expert_outputs.shape[1]), device=expert_outputs.device, dtype=dtype
    )
    output.index_add_(0, routes.token_ids, weighted)
    return output.to(expert_outputs.dtype)
