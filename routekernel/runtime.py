"""Readable inference backends for a bias-free, two-layer ReLU MoE."""

from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import Tensor

from .routing import (
    _compile_routes, accumulation_dtype, merge_outputs, validate_routes,
)


@dataclass(frozen=True)
class ExpertWeights:
    up: Tensor    # [experts, input_features, hidden_features]
    down: Tensor  # [experts, hidden_features, input_features]


def expert_mlp(tokens: Tensor, up: Tensor, down: Tensor) -> Tensor:
    """Expand, remove negative activations, then project back to input width."""
    return torch.relu(tokens @ up) @ down


def validate_inputs(tokens: Tensor, ids: Tensor, weights: Tensor, experts: ExpertWeights) -> None:
    if tokens.ndim != 2 or not tokens.is_floating_point() or tokens.shape[1] < 1:
        raise ValueError("tokens must be a floating-point [T, D] tensor with D > 0")
    if experts.up.ndim != 3 or experts.down.ndim != 3:
        raise ValueError("expert matrices must have shapes [E, D, H] and [E, H, D]")
    e, d, h = experts.up.shape
    if h < 1 or d != tokens.shape[1] or experts.down.shape != (e, h, d):
        raise ValueError("expert matrix dimensions do not match tokens")
    for matrix in (experts.up, experts.down):
        if matrix.device != tokens.device or matrix.dtype != tokens.dtype:
            raise ValueError("expert matrices and tokens must share dtype and device")
    validate_routes(ids, weights, e)
    if ids.shape[0] != tokens.shape[0] or ids.device != tokens.device:
        raise ValueError("routes must match the token count and device")


@torch.inference_mode()
def reference_forward(tokens: Tensor, ids: Tensor, weights: Tensor, experts: ExpertWeights) -> Tensor:
    """Correctness oracle: explicitly execute each token-expert assignment."""
    validate_inputs(tokens, ids, weights, experts)
    acc_dtype = accumulation_dtype(tokens.dtype)
    output = torch.zeros(tokens.shape, device=tokens.device, dtype=acc_dtype)
    for token in range(tokens.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot].item())
            value = expert_mlp(tokens[token], experts.up[expert], experts.down[expert])
            output[token] += weights[token, slot].to(acc_dtype) * value.to(acc_dtype)
    return output.to(tokens.dtype)


def _pytorch_forward(tokens: Tensor, ids: Tensor, weights: Tensor, experts: ExpertWeights) -> Tensor:
    """Practical eager baseline: find and batch each expert's assigned tokens."""
    acc_dtype = accumulation_dtype(tokens.dtype)
    output = torch.zeros(tokens.shape, device=tokens.device, dtype=acc_dtype)
    for expert in range(experts.up.shape[0]):
        token_ids, slots = torch.where(ids == expert)
        if token_ids.numel() == 0:
            continue
        values = expert_mlp(tokens[token_ids], experts.up[expert], experts.down[expert])
        weighted = values.to(acc_dtype) * weights[token_ids, slots].to(acc_dtype).unsqueeze(1)
        output.index_add_(0, token_ids, weighted)
    return output.to(tokens.dtype)


@lru_cache(maxsize=1)
def _compiled_mlp():
    # Compile the dense expert arithmetic. Routing and the Python expert loop
    # remain eager; this scope is explicitly recorded in benchmark reports.
    return torch.compile(expert_mlp, fullgraph=True, dynamic=True)


def _grouped_forward(
    tokens: Tensor, ids: Tensor, weights: Tensor, experts: ExpertWeights, backend: str
) -> Tensor:
    routes = _compile_routes(ids, weights, experts.up.shape[0])
    grouped_tokens = tokens[routes.token_ids].contiguous()
    if backend == "triton":
        from .triton_backend import grouped_mlp
        values = grouped_mlp(grouped_tokens, experts, routes.expert_offsets)
    else:
        values = torch.empty_like(grouped_tokens)
        mlp = _compiled_mlp() if backend == "torch-compiled" else expert_mlp
        # One device-to-host transfer for all boundaries. This is a known GPU
        # bottleneck in the readable PyTorch grouped backend, included in timing.
        offsets = routes.expert_offsets.tolist()
        for expert, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            if start != end:
                values[start:end] = mlp(
                    grouped_tokens[start:end], experts.up[expert], experts.down[expert]
                )
    return merge_outputs(values, routes)


BACKENDS = ("pytorch", "grouped", "torch-compiled", "triton")


def _execute(tokens: Tensor, ids: Tensor, weights: Tensor, experts: ExpertWeights, backend: str) -> Tensor:
    """Internal entry for benchmarks that validate their fixed inputs once."""
    if backend == "pytorch":
        return _pytorch_forward(tokens, ids, weights, experts)
    if backend in ("grouped", "torch-compiled", "triton"):
        return _grouped_forward(tokens, ids, weights, experts, backend)
    raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")


@torch.inference_mode()
def forward(
    tokens: Tensor, ids: Tensor, weights: Tensor, experts: ExpertWeights, backend: str = "grouped"
) -> Tensor:
    """Execute supplied routing decisions; this function does not run a router."""
    validate_inputs(tokens, ids, weights, experts)
    return _execute(tokens, ids, weights, experts, backend)
