"""
Fused MoE Kernel for FlashInfer Competition.

Track: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
Model: DeepSeek-V3/R1

V007: Fused FP8-A GEMM + autotune.
- GEMM1: loads hidden_states as FP8 directly, fuses dequant of both A and B in tile loop
  - Eliminates separate _dequant_hidden_kernel entirely
  - 4x less memory traffic for A (FP8 vs FP32)
- GEMM2: keeps FP32 A (SwiGLU output) with fused B dequant (same as V006)
- Autotune on both GEMM kernels (BLOCK_M, num_warps, num_stages)
- Batch .tolist() + pre-allocated buffers from V006
"""

import torch
import triton
import triton.language as tl

# ── Constants ──────────────────────────────────────────────────────
BLOCK = 128
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
E_GLOBAL = 256


# ── Triton kernels ──────────────────────────────────────────────────

# Autotune configs (shared by both GEMM kernels)
_gemm_configs = [
    triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=4),
]


@triton.autotune(configs=_gemm_configs, key=['N', 'K'])
@triton.jit
def _fp8_dual_dequant_gemm_kernel(
    # A: FP8 with per-row block scale
    A_ptr, A_scale_ptr,
    # B: FP8 with per-block scale
    B_ptr, B_scale_ptr,
    # Output
    C_ptr,
    # Dimensions
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    # Strides
    stride_am,
    stride_bn,
    # Tile size (autotuned)
    BLOCK_M: tl.constexpr,
):
    """Fused dual-dequant GEMM: C[M,N] = A[M,K](fp8) @ B[N,K](fp8).T

    Both A and B are FP8 with block-scale quantization (block=128).
    A_scale layout: [M, K//128] row-major (pre-transposed from original [K//128, T]).
    B_scale layout: [N//128, K//128] row-major.
    FP8 dot product (FP8 tensor cores), scales applied after dot.
    """
    BLOCK_N: tl.constexpr = 128
    BLOCK_K: tl.constexpr = 128
    NUM_K_BLOCKS: tl.constexpr = K // BLOCK_K

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    # A pointers: [BLOCK_M, BLOCK_K]
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :]

    # B^T pointers: [BLOCK_K, BLOCK_N] (B is [N,K] row-major)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bt_ptrs = B_ptr + offs_n[None, :] * stride_bn + tl.arange(0, BLOCK_K)[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M

    for k_step in tl.static_range(NUM_K_BLOCKS):
        # Load A tile as FP8
        a_fp8 = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        # Load B^T tile as FP8
        bt_fp8 = tl.load(bt_ptrs)

        # FP8 × FP8 dot → FP32 accumulator (FP8 tensor cores)
        raw = tl.dot(a_fp8, bt_fp8)

        # Apply dual scales after dot (both are constant within this 128-block)
        scale_a = tl.load(A_scale_ptr + offs_m * NUM_K_BLOCKS + k_step,
                          mask=mask_m, other=0.0)
        scale_b = tl.load(B_scale_ptr + pid_n * NUM_K_BLOCKS + k_step)
        acc += raw * scale_a[:, None] * scale_b

        # Advance K pointers
        a_ptrs += BLOCK_K
        bt_ptrs += BLOCK_K

    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * N + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    tl.store(c_ptrs, acc, mask=mask_m[:, None])


@triton.autotune(configs=_gemm_configs, key=['N', 'K'])
@triton.jit
def _fused_dequant_gemm_kernel(
    # A: FP32 input (from SwiGLU)
    A_ptr, B_ptr, B_scale_ptr, C_ptr,
    # Dimensions
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    # Strides
    stride_am,
    stride_bn,
    # Scale layout
    num_scale_cols: tl.constexpr,
    # Tile size (autotuned)
    BLOCK_M: tl.constexpr,
):
    """Fused dequant GEMM: C[M,N] = A[M,K](f32) @ B[N,K](fp8).T

    A is FP32 (SwiGLU output). B is FP8, dequanted to FP32.
    TF32 tensor cores (2250 TFLOPS on B200), FP32 accumulator.
    """
    BLOCK_N: tl.constexpr = 128
    BLOCK_K: tl.constexpr = 128

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :]

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bt_ptrs = B_ptr + offs_n[None, :] * stride_bn + tl.arange(0, BLOCK_K)[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M

    for k_step in tl.static_range(K // BLOCK_K):
        # Load A as FP32 (SwiGLU output)
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)

        # Load B as FP8, dequant to FP32
        bt_fp8 = tl.load(bt_ptrs)
        scale = tl.load(B_scale_ptr + pid_n * num_scale_cols + k_step)
        bt_f32 = bt_fp8.to(tl.float32) * scale

        # TF32 dot → FP32 accumulator
        acc += tl.dot(a, bt_f32)

        a_ptrs += BLOCK_K
        bt_ptrs += BLOCK_K

    c_ptrs = C_ptr + offs_m[:, None] * N + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    tl.store(c_ptrs, acc, mask=mask_m[:, None])


@triton.jit
def _swiglu_kernel(
    input_ptr, out_ptr,
    N, I: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    """SwiGLU: silu(gate) * up. FP32 in, FP32 out."""
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    for block_idx in tl.static_range((I + BLOCK_SIZE - 1) // BLOCK_SIZE):
        cols = block_idx * BLOCK_SIZE + col_offsets
        mask = cols < I

        up = tl.load(input_ptr + row * 2 * I + cols, mask=mask, other=0.0)
        gate = tl.load(input_ptr + row * 2 * I + I + cols, mask=mask, other=0.0)

        silu_gate = gate * tl.sigmoid(gate)
        result = silu_gate * up

        tl.store(out_ptr + row * I + cols, result, mask=mask)


# ── Helper functions ────────────────────────────────────────────────

def _fp8_dual_dequant_gemm(A_fp8, A_scale, B_fp8, B_scale, M, N, K, out=None):
    """GEMM1: A(FP8) with row-scale × B(FP8) with block-scale → C(FP32)."""
    C = out if out is not None else torch.empty((M, N), dtype=torch.float32, device=A_fp8.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, BLOCK))
    _fp8_dual_dequant_gemm_kernel[grid](
        A_fp8, A_scale, B_fp8, B_scale, C,
        M, N, K,
        A_fp8.stride(0), B_fp8.stride(0),
    )
    return C


def _fused_dequant_gemm(A, B_fp8, B_scale, M, N, K, out=None):
    """GEMM2: A(FP32) × B(FP8) with block-scale → C(FP32)."""
    C = out if out is not None else torch.empty((M, N), dtype=torch.float32, device=A.device)
    num_scale_cols = K // BLOCK
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, BLOCK))
    _fused_dequant_gemm_kernel[grid](
        A, B_fp8, B_scale, C,
        M, N, K,
        A.stride(0), B_fp8.stride(0),
        num_scale_cols,
    )
    return C


def _swiglu(gemm1_out, N, I, out=None):
    """SwiGLU on FP32 input -> FP32 output [N, I]."""
    result = out if out is not None else torch.empty((N, I), dtype=torch.float32, device=gemm1_out.device)
    block_size = min(BLOCK, I)
    _swiglu_kernel[(N,)](gemm1_out, result, N, I, block_size, num_warps=4)
    return result


def _deepseek_v3_routing(routing_logits, routing_bias, routed_scaling_factor):
    """DeepSeek-V3 no-aux routing."""
    T = routing_logits.shape[0]
    device = routing_logits.device

    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)

    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    group_size = E_GLOBAL // N_GROUP
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)

    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros(T, N_GROUP, device=device)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask
        .unsqueeze(2)
        .expand(T, N_GROUP, group_size)
        .reshape(T, E_GLOBAL)
    )

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    expert_mask = torch.zeros(T, E_GLOBAL, device=device)
    expert_mask.scatter_(1, topk_idx, 1.0)
    weights = s * expert_mask
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    return topk_idx, weights


# ── Main entry point (DPS) ──────────────────────────────────────────

def kernel(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
    output,
):
    """Fused MoE kernel: sorted dispatch + fused FP8-A GEMM (V007).

    Key changes from V006:
    1. GEMM1 loads A as FP8 directly — no separate dequant kernel
    2. Dual block-scale dequant fused in GEMM1 tile loop (A_scale + B_scale)
    3. 4x less memory traffic for A reads (FP8 vs FP32)
    4. GEMM2 unchanged (FP32 SwiGLU output × FP8 weights)
    """
    device = hidden_states.device
    T = routing_logits.shape[0]
    E_local = gemm1_weights.shape[0]
    H = hidden_states.shape[1]
    I = gemm2_weights.shape[2]
    gemm1_out_size = gemm1_weights.shape[1]

    local_start = local_expert_offset.item() if isinstance(local_expert_offset, torch.Tensor) else int(local_expert_offset)
    scaling = routed_scaling_factor.item() if isinstance(routed_scaling_factor, torch.Tensor) else float(routed_scaling_factor)

    # ── Part A: Routing ──
    topk_idx, weights = _deepseek_v3_routing(
        routing_logits, routing_bias, scaling,
    )

    # ── Part B: Sorted dispatch ──
    token_ids = torch.arange(T, device=device).unsqueeze(1).expand(T, TOP_K).reshape(-1)
    expert_ids = topk_idx.reshape(-1)
    local_expert_ids = expert_ids - local_start

    valid_mask = (local_expert_ids >= 0) & (local_expert_ids < E_local)
    valid_token_ids = token_ids[valid_mask]
    valid_local_expert_ids = local_expert_ids[valid_mask]
    valid_global_expert_ids = expert_ids[valid_mask]

    if valid_token_ids.numel() == 0:
        output.zero_()
        return

    sorted_indices = torch.argsort(valid_local_expert_ids, stable=True)
    sorted_token_ids = valid_token_ids[sorted_indices]
    sorted_local_eids = valid_local_expert_ids[sorted_indices]
    sorted_global_eids = valid_global_expert_ids[sorted_indices]

    unique_experts, expert_counts = torch.unique_consecutive(sorted_local_eids, return_counts=True)

    # ── Part C: Gather FP8 hidden_states + scales (no dequant!) ──
    # FP8 data: [N_sorted, H] — 4x smaller than FP32
    A_fp8_sorted = hidden_states[sorted_token_ids]
    # Scale: original [NUM_H_BLOCKS, T] → gather+transpose → [N_sorted, NUM_H_BLOCKS]
    A_scale_sorted = hidden_states_scale.T[sorted_token_ids]

    w_sorted = weights[sorted_token_ids, sorted_global_eids]

    result = torch.zeros((T, H), dtype=torch.float32, device=device)

    # ── Batch .item() → .tolist() ──
    unique_experts_list = unique_experts.tolist()
    expert_counts_list = expert_counts.tolist()

    # ── Pre-allocate output buffers ──
    max_count = max(expert_counts_list)
    G1_buf = torch.empty((max_count, gemm1_out_size), dtype=torch.float32, device=device)
    C_buf = torch.empty((max_count, I), dtype=torch.float32, device=device)
    O_buf = torch.empty((max_count, H), dtype=torch.float32, device=device)

    # ── Process each expert ──
    offset = 0
    for i in range(len(unique_experts_list)):
        le = int(unique_experts_list[i])
        count = int(expert_counts_list[i])

        tids = sorted_token_ids[offset:offset + count]
        w_e = w_sorted[offset:offset + count]

        # A for this expert: FP8 + scale
        A_e_fp8 = A_fp8_sorted[offset:offset + count]      # [count, H] FP8
        A_e_scale = A_scale_sorted[offset:offset + count]   # [count, NUM_H_BLOCKS] FP32

        # GEMM1: FP8 A × FP8 B with dual dequant → [count, 2I] FP32
        G1 = _fp8_dual_dequant_gemm(
            A_e_fp8, A_e_scale,
            gemm1_weights[le], gemm1_weights_scale[le],
            count, gemm1_out_size, H,
            out=G1_buf[:count],
        )

        # SwiGLU: FP32 → FP32
        C = _swiglu(G1, count, I, out=C_buf[:count])

        # GEMM2: FP32 A × FP8 B with B dequant → [count, H] FP32
        O = _fused_dequant_gemm(
            C, gemm2_weights[le], gemm2_weights_scale[le],
            count, H, I,
            out=O_buf[:count],
        )

        # Weighted accumulation
        result.index_add_(0, tids, O * w_e.unsqueeze(1))

        offset += count

    output.copy_(result.to(torch.bfloat16))
