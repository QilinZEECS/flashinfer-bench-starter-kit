"""
Fused MoE Kernel for FlashInfer Competition.

Track: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
Model: DeepSeek-V3/R1

V008: Grouped GEMM — eliminates Python expert loop.
- 3D Triton grid (experts x M_tiles x N_tiles) processes all experts in one launch
- 3 kernel launches (GEMM1 + SwiGLU + GEMM2) + 1 index_add_ replaces ~128 launches
- Eliminates per-expert Python overhead and kernel launch latency
- FP8 tensor cores for GEMM1, TF32 for GEMM2
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

_gemm_configs = [
    triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=4),
]


@triton.autotune(configs=_gemm_configs, key=['N', 'K'])
@triton.jit
def _grouped_fp8_dual_dequant_gemm_kernel(
    A_ptr, A_scale_ptr,
    B_ptr, B_scale_ptr,
    C_ptr,
    expert_offsets_ptr, expert_ids_ptr,
    N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_be, stride_bn, stride_bse,
    stride_cm,
    BLOCK_M: tl.constexpr,
):
    """Grouped GEMM1: C = A(fp8) @ B(fp8).T with dual block-scale dequant.

    3D grid: (num_experts, max_m_tiles, N // 128).
    FP8 tensor cores, scales applied after dot.
    """
    BLOCK_N: tl.constexpr = 128
    BLOCK_K: tl.constexpr = 128
    NUM_K_BLOCKS: tl.constexpr = K // BLOCK_K

    eidx = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    row_start = tl.load(expert_offsets_ptr + eidx)
    row_end = tl.load(expert_offsets_ptr + eidx + 1)
    count = row_end - row_start

    if pid_m * BLOCK_M >= count:
        return

    expert_id = tl.load(expert_ids_ptr + eidx)

    offs_m = row_start + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    local_offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = local_offs_m < count
    offs_k = tl.arange(0, BLOCK_K)

    # A pointers: [BLOCK_M, BLOCK_K]
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :]

    # B^T pointers for this expert: [BLOCK_K, BLOCK_N]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_base = B_ptr + expert_id * stride_be
    bt_ptrs = b_base + offs_n[None, :] * stride_bn + offs_k[:, None]

    # B_scale base for this expert
    bscale_base = B_scale_ptr + expert_id * stride_bse

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_step in tl.static_range(NUM_K_BLOCKS):
        a_fp8 = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        bt_fp8 = tl.load(bt_ptrs)

        # FP8 x FP8 dot -> FP32 (FP8 tensor cores)
        raw = tl.dot(a_fp8, bt_fp8)

        # Dual scale: A_scale[row, k_block] * B_scale[n_block, k_block]
        scale_a = tl.load(A_scale_ptr + offs_m * NUM_K_BLOCKS + k_step,
                          mask=mask_m, other=0.0)
        scale_b = tl.load(bscale_base + pid_n * NUM_K_BLOCKS + k_step)
        acc += raw * scale_a[:, None] * scale_b

        a_ptrs += BLOCK_K
        bt_ptrs += BLOCK_K

    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    tl.store(c_ptrs, acc, mask=mask_m[:, None])


@triton.autotune(configs=_gemm_configs, key=['N', 'K'])
@triton.jit
def _grouped_fused_dequant_gemm_kernel(
    A_ptr,
    B_ptr, B_scale_ptr,
    C_ptr,
    expert_offsets_ptr, expert_ids_ptr,
    N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_be, stride_bn, stride_bse,
    stride_cm,
    BLOCK_M: tl.constexpr,
):
    """Grouped GEMM2: C = A(f32) @ B(fp8).T with B block-scale dequant.

    3D grid: (num_experts, max_m_tiles, N // 128).
    TF32 tensor cores, FP32 accumulator.
    """
    BLOCK_N: tl.constexpr = 128
    BLOCK_K: tl.constexpr = 128
    NUM_K_BLOCKS: tl.constexpr = K // BLOCK_K

    eidx = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    row_start = tl.load(expert_offsets_ptr + eidx)
    row_end = tl.load(expert_offsets_ptr + eidx + 1)
    count = row_end - row_start

    if pid_m * BLOCK_M >= count:
        return

    expert_id = tl.load(expert_ids_ptr + eidx)

    offs_m = row_start + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    local_offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = local_offs_m < count
    offs_k = tl.arange(0, BLOCK_K)

    # A pointers: FP32 [BLOCK_M, BLOCK_K]
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :]

    # B^T pointers for this expert: FP8 [BLOCK_K, BLOCK_N]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_base = B_ptr + expert_id * stride_be
    bt_ptrs = b_base + offs_n[None, :] * stride_bn + offs_k[:, None]

    # B_scale base for this expert
    bscale_base = B_scale_ptr + expert_id * stride_bse

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_step in tl.static_range(NUM_K_BLOCKS):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)

        # Load B as FP8, dequant to FP32
        bt_fp8 = tl.load(bt_ptrs)
        scale = tl.load(bscale_base + pid_n * NUM_K_BLOCKS + k_step)
        bt_f32 = bt_fp8.to(tl.float32) * scale

        # TF32 dot -> FP32 accumulator
        acc += tl.dot(a, bt_f32)

        a_ptrs += BLOCK_K
        bt_ptrs += BLOCK_K

    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
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

def _grouped_fp8_dual_dequant_gemm(A_fp8, A_scale, B_fp8, B_scale,
                                    expert_offsets, expert_ids,
                                    max_count, N, K, out):
    """Grouped GEMM1: all experts in single launch."""
    num_experts = expert_ids.shape[0]
    grid = lambda META: (
        num_experts,
        triton.cdiv(max_count, META['BLOCK_M']),
        N // BLOCK,
    )
    _grouped_fp8_dual_dequant_gemm_kernel[grid](
        A_fp8, A_scale,
        B_fp8, B_scale,
        out,
        expert_offsets, expert_ids,
        N, K,
        A_fp8.stride(0),
        B_fp8.stride(0), B_fp8.stride(1),
        B_scale.stride(0),
        out.stride(0),
    )


def _grouped_fused_dequant_gemm(A, B_fp8, B_scale,
                                 expert_offsets, expert_ids,
                                 max_count, N, K, out):
    """Grouped GEMM2: all experts in single launch."""
    num_experts = expert_ids.shape[0]
    grid = lambda META: (
        num_experts,
        triton.cdiv(max_count, META['BLOCK_M']),
        N // BLOCK,
    )
    _grouped_fused_dequant_gemm_kernel[grid](
        A,
        B_fp8, B_scale,
        out,
        expert_offsets, expert_ids,
        N, K,
        A.stride(0),
        B_fp8.stride(0), B_fp8.stride(1),
        B_scale.stride(0),
        out.stride(0),
    )


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
    """Fused MoE kernel: grouped GEMM (V008).

    Key changes from V007:
    1. Eliminates Python expert loop — all experts in single kernel launch
    2. 3D Triton grid (experts x M_tiles x N_tiles) with per-expert early exit
    3. 3 kernel launches + 1 index_add_ replaces ~128 per-expert launches
    4. Reduces kernel launch overhead and Python-side synchronization
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

    # ── Part C: Build dispatch table ──
    unique_experts, expert_counts = torch.unique_consecutive(sorted_local_eids, return_counts=True)
    expert_offsets = torch.zeros(unique_experts.shape[0] + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = expert_counts.cumsum(0)
    N_sorted = sorted_token_ids.shape[0]
    max_count = expert_counts.max().item()

    # ── Part D: Gather FP8 hidden_states + scales ──
    A_fp8_sorted = hidden_states[sorted_token_ids]
    A_scale_sorted = hidden_states_scale.T[sorted_token_ids]
    w_sorted = weights[sorted_token_ids, sorted_global_eids]

    # ── Part E: Grouped GEMM1 + SwiGLU + GEMM2 ──
    G1_buf = torch.empty((N_sorted, gemm1_out_size), dtype=torch.float32, device=device)
    C_buf = torch.empty((N_sorted, I), dtype=torch.float32, device=device)
    O_buf = torch.empty((N_sorted, H), dtype=torch.float32, device=device)

    # GEMM1: FP8 A x FP8 B -> FP32 (all experts, single launch)
    _grouped_fp8_dual_dequant_gemm(
        A_fp8_sorted, A_scale_sorted,
        gemm1_weights, gemm1_weights_scale,
        expert_offsets, unique_experts,
        max_count, gemm1_out_size, H,
        out=G1_buf,
    )

    # SwiGLU: FP32 -> FP32 (all rows, single launch)
    _swiglu(G1_buf, N_sorted, I, out=C_buf)

    # GEMM2: FP32 A x FP8 B -> FP32 (all experts, single launch)
    _grouped_fused_dequant_gemm(
        C_buf,
        gemm2_weights, gemm2_weights_scale,
        expert_offsets, unique_experts,
        max_count, H, I,
        out=O_buf,
    )

    # ── Part F: Weighted scatter (single operation) ──
    result = torch.zeros((T, H), dtype=torch.float32, device=device)
    result.index_add_(0, sorted_token_ids, O_buf * w_sorted.unsqueeze(1))

    output.copy_(result.to(torch.bfloat16))
