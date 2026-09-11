"""Experimental CUDA grouped GEMM backend; requires GPU validation.

Two kernel launches evaluate the two expert projections. Routes, gathering,
and weighted merging still use PyTorch. No GPU performance claim is made.
"""

import sys

import torch


def require_triton() -> None:
    if sys.platform != "linux":
        raise RuntimeError("The Triton backend requires Linux/WSL with a supported NVIDIA GPU")
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("The Triton backend requires CUDA-enabled PyTorch and an NVIDIA GPU")
    try:
        import triton  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install RouteKernel's gpu extra after installing CUDA PyTorch") from exc


# Import lazily: CPU demos and benchmarks must not require Triton.
def _make_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def grouped_projection(
        X, W, Y, OFFSETS, TILE_OFFSETS,
        E: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        EXPERT_BLOCK: tl.constexpr, RELU: tl.constexpr,
    ):
        tile = tl.program_id(0)
        col_tile = tl.program_id(1)
        total_tiles = tl.load(TILE_OFFSETS + E)
        if tile < total_tiles:
            # Locate the expert that owns this row tile. Equal boundaries skip
            # unused experts. No host read of the routing distribution is needed.
            expert_range = tl.arange(0, EXPERT_BLOCK)
            starts = tl.load(TILE_OFFSETS + expert_range, expert_range < E, other=2147483647)
            expert = tl.sum(((starts <= tile) & (expert_range < E)).to(tl.int32), axis=0) - 1
            # Large expert parameter banks can exceed 2^31 elements.
            expert = expert.to(tl.int64)
            tile_start = tl.load(TILE_OFFSETS + expert)
            row_start = tl.load(OFFSETS + expert)
            row_end = tl.load(OFFSETS + expert + 1)
            rows = row_start + (tile - tile_start) * BM + tl.arange(0, BM)
            cols = col_tile * BN + tl.arange(0, BN)
            inner = tl.arange(0, BK)
            accumulator = tl.zeros((BM, BN), tl.float32)
            for chunk in range(tl.cdiv(K, BK)):
                kk = chunk * BK + inner
                a = tl.load(
                    X + rows[:, None] * K + kk[None, :],
                    (rows[:, None] < row_end) & (kk[None, :] < K), other=0.0,
                )
                b = tl.load(
                    W + expert * K * N + kk[:, None] * N + cols[None, :],
                    (kk[:, None] < K) & (cols[None, :] < N), other=0.0,
                )
                accumulator += tl.dot(a, b, input_precision="ieee")
            if RELU:
                accumulator = tl.maximum(accumulator, 0.0)
            tl.store(
                Y + rows[:, None] * N + cols[None, :], accumulator,
                (rows[:, None] < row_end) & (cols[None, :] < N),
            )

    return grouped_projection


_kernel = None


def grouped_mlp(tokens, experts, offsets):
    require_triton()
    import triton

    if tokens.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Triton supports float16, bfloat16, and float32 inputs")
    if tokens.shape[0] == 0:
        return torch.empty_like(tokens)
    global _kernel
    if _kernel is None:
        _kernel = _make_kernel()
    tokens = tokens.contiguous()
    up, down = experts.up.contiguous(), experts.down.contiguous()
    e, d, h = up.shape
    bm, bn, bk = 32, 32, 32
    counts = offsets[1:] - offsets[:-1]
    tile_counts = (counts + bm - 1) // bm
    tile_offsets = torch.cat((tile_counts.new_zeros(1), tile_counts.cumsum(0)))
    # ceil(sum(counts)/BM) + E - 1 bounds sum(ceil(counts/BM)). Excess
    # programs exit in the kernel. This avoids a GPU-to-CPU scalar transfer.
    row_tiles = triton.cdiv(tokens.shape[0], bm) + e - 1
    hidden = torch.empty((tokens.shape[0], h), device=tokens.device, dtype=tokens.dtype)
    output = torch.empty_like(tokens)
    for x, w, y, k, n, relu in (
        (tokens, up, hidden, d, h, True), (hidden, down, output, h, d, False),
    ):
        _kernel[(row_tiles, triton.cdiv(n, bn))](
            x, w, y, offsets, tile_offsets, E=e, K=k, N=n,
            BM=bm, BN=bn, BK=bk, EXPERT_BLOCK=triton.next_power_of_2(e),
            RELU=relu, num_warps=4,
        )
    return output
