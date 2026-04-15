"""
Fused MoE Kernel for FlashInfer Competition.

Track: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
Model: DeepSeek-V3/R1

V012: Fused dispatch kernel — replaces ~15 PyTorch dispatch ops with single Triton kernel.
- Histogram + prefix-sum + scatter replaces nonzero, argsort, unique_consecutive
- Eliminates 7 DtoH syncs (item/nonzero) → single meta.tolist()
- Routing + GEMM architecture unchanged from V009
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
def _grouped_fp16_fp8_gemm2_kernel(
    A_ptr, A_scale_ptr,
    B_ptr, B_scale_ptr,
    C_ptr,
    expert_offsets_ptr, expert_ids_ptr,
    weights_ptr,
    N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_be, stride_bn, stride_bse,
    stride_cm,
    BLOCK_M: tl.constexpr,
):
    """Grouped GEMM2: C = A(fp16, pre-normalized) @ B(fp8).T with dual block-scale.

    A is FP16 pre-normalized per 128-block, A_scale has the magnitude.
    B is FP8 with block scale. FP16 tensor cores (2x TF32 throughput).
    Fuses per-row weight multiply into store epilogue.
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

    # A pointers: FP16 [BLOCK_M, BLOCK_K] (pre-normalized)
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :]

    # B^T pointers for this expert: FP8 [BLOCK_K, BLOCK_N]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_base = B_ptr + expert_id * stride_be
    bt_ptrs = b_base + offs_n[None, :] * stride_bn + offs_k[:, None]

    # B_scale base for this expert
    bscale_base = B_scale_ptr + expert_id * stride_bse

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_step in tl.static_range(NUM_K_BLOCKS):
        a_fp16 = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)

        # Load B as FP8 → cast to FP16 for FP16 tensor cores
        bt_fp8 = tl.load(bt_ptrs)
        bt_fp16 = bt_fp8.to(tl.float16)

        # FP16 x FP16 dot -> FP32
        raw = tl.dot(a_fp16, bt_fp16)

        # Dual scale: A_scale[row, k_block] * B_scale[n_block, k_block]
        scale_a = tl.load(A_scale_ptr + offs_m * NUM_K_BLOCKS + k_step,
                          mask=mask_m, other=0.0)
        scale_b = tl.load(bscale_base + pid_n * NUM_K_BLOCKS + k_step)
        acc += raw * scale_a[:, None] * scale_b

        a_ptrs += BLOCK_K
        bt_ptrs += BLOCK_K

    # Fuse per-row weight multiply into store
    w = tl.load(weights_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc * w[:, None]

    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    tl.store(c_ptrs, acc, mask=mask_m[:, None])


@triton.jit
def _routing_kernel(
    logits_ptr, bias_ptr,
    topk_idx_ptr, topk_weights_ptr,
    routed_scaling_factor,
    stride_logits,
    E_GLOBAL: tl.constexpr,
    N_GROUP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    TOP_K: tl.constexpr,
):
    """Fused DeepSeek-V3 routing: sigmoid + group scoring + topk selection.

    Grid: (T,). Each program handles one token.
    Outputs sparse topk_idx[T, TOP_K] and topk_weights[T, TOP_K].
    """
    t = tl.program_id(0)
    NEG_INF = float('-inf')
    offs_e = tl.arange(0, E_GLOBAL)

    # Step 1: sigmoid + bias
    logits = tl.load(logits_ptr + t * stride_logits + offs_e).to(tl.float32)
    bias = tl.load(bias_ptr + offs_e).to(tl.float32)
    s = tl.sigmoid(logits)
    s_with_bias = s + bias

    expert_group = offs_e // GROUP_SIZE  # [E_GLOBAL] group id per expert

    # Step 2: Group scoring — top-2 per group via masked max on [E_GLOBAL]
    offs_g = tl.arange(0, N_GROUP)
    group_scores = tl.zeros((N_GROUP,), dtype=tl.float32)
    for g in tl.static_range(N_GROUP):
        g_vals = tl.where(expert_group == g, s_with_bias, NEG_INF)
        top1_idx = tl.argmax(g_vals, axis=0)
        top1_val = tl.max(g_vals, axis=0)
        g_vals2 = tl.where(offs_e == top1_idx, NEG_INF, g_vals)
        top2_val = tl.max(g_vals2, axis=0)
        top2_val = tl.where(top2_val == NEG_INF, 0.0, top2_val)
        group_scores = tl.where(offs_g == g, top1_val + top2_val, group_scores)

    # Step 3: Top-4 groups → directly mark valid experts in [E_GLOBAL]
    expert_valid = tl.zeros((E_GLOBAL,), dtype=tl.float32)
    gs_work = group_scores
    for _k in tl.static_range(TOPK_GROUP):
        g_idx = tl.argmax(gs_work, axis=0)
        expert_valid = tl.where(expert_group == g_idx, 1.0, expert_valid)
        gs_work = tl.where(offs_g == g_idx, NEG_INF, gs_work)

    # Step 4: Top-8 experts from valid set
    scores_pruned = tl.where(expert_valid > 0.0, s_with_bias, NEG_INF)
    for k in tl.static_range(TOP_K):
        best_idx = tl.argmax(scores_pruned, axis=0)
        tl.store(topk_idx_ptr + t * TOP_K + k, best_idx.to(tl.int64))
        # Extract unbiased sigmoid at selected index for weight computation
        s_val = tl.sum(tl.where(offs_e == best_idx, s, 0.0))
        tl.store(topk_weights_ptr + t * TOP_K + k, s_val)
        scores_pruned = tl.where(offs_e == best_idx, NEG_INF, scores_pruned)

    # Step 5: Normalize weights
    w_offs = tl.arange(0, TOP_K)
    w_vals = tl.load(topk_weights_ptr + t * TOP_K + w_offs)
    w_sum = tl.sum(w_vals, axis=0) + 1e-20
    w_norm = (w_vals / w_sum) * routed_scaling_factor
    tl.store(topk_weights_ptr + t * TOP_K + w_offs, w_norm)


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


@triton.jit
def _dispatch_kernel(
    topk_idx_ptr, topk_weights_ptr,
    local_start,
    sorted_token_ids_ptr, sorted_weights_ptr,
    expert_offsets_ptr,
    meta_ptr,
    TOTAL,
    TOP_K: tl.constexpr,
    E_LOCAL: tl.constexpr,
):
    """Fused dispatch: histogram + prefix-sum + scatter in one kernel.

    Dense layout: expert_offsets[E_LOCAL+1] covers all 32 experts.
    GEMM kernels early-exit for empty experts (count=0).
    Grid: (1,). Single program processes all T*TOP_K entries.
    """
    offs_e = tl.arange(0, E_LOCAL)

    # Step 1: Histogram — count tokens per local expert
    counts = tl.zeros((E_LOCAL,), dtype=tl.int32)
    for i in range(TOTAL):
        eid = tl.load(topk_idx_ptr + i).to(tl.int32)
        local_eid = eid - local_start
        valid = (local_eid >= 0) & (local_eid < E_LOCAL)
        counts += tl.where((offs_e == local_eid) & valid, 1, 0)

    # Step 2: Exclusive prefix sum → expert_offsets[0..E_LOCAL]
    # offsets[e] = sum(counts[0..e-1])
    offsets = tl.zeros((E_LOCAL,), dtype=tl.int32)
    for e in tl.static_range(1, E_LOCAL):
        # offsets[e] = sum of counts[0..e-1]
        prev_sum = tl.sum(tl.where(offs_e < e, counts, 0))
        offsets = tl.where(offs_e == e, prev_sum, offsets)
    N_valid = tl.sum(counts)

    # Store dense expert_offsets[E_LOCAL+1]
    tl.store(expert_offsets_ptr + offs_e, offsets.to(tl.int64))
    tl.store(expert_offsets_ptr + E_LOCAL, N_valid.to(tl.int64))

    # Find max_count
    max_count = tl.max(counts, axis=0)

    # Store metadata: [max_count, N_valid]
    tl.store(meta_ptr + 0, max_count.to(tl.int64))
    tl.store(meta_ptr + 1, N_valid.to(tl.int64))

    # Step 3: Scatter tokens to sorted positions
    write_pos = offsets  # per-expert write cursor (mutable copy)
    for i in range(TOTAL):
        eid = tl.load(topk_idx_ptr + i).to(tl.int32)
        local_eid = eid - local_start
        valid = (local_eid >= 0) & (local_eid < E_LOCAL)
        # Branchless scatter: compute pos, only store if valid
        pos = tl.sum(tl.where(offs_e == local_eid, write_pos, 0))
        token_id = i // TOP_K
        if valid:
            tl.store(sorted_token_ids_ptr + pos, token_id.to(tl.int64))
            tl.store(sorted_weights_ptr + pos, tl.load(topk_weights_ptr + i))
        # Update write cursor (branchless: only the matching expert increments)
        write_pos += tl.where((offs_e == local_eid) & valid, 1, 0)


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


def _grouped_fp16_gemm2(A_fp16, A_scale, B_fp8, B_scale,
                        expert_offsets, expert_ids,
                        weights, max_count, N, K, out):
    """Grouped GEMM2 with FP16×FP8 tensor cores + fused weight multiply."""
    num_experts = expert_ids.shape[0]
    grid = lambda META: (
        num_experts,
        triton.cdiv(max_count, META['BLOCK_M']),
        N // BLOCK,
    )
    _grouped_fp16_fp8_gemm2_kernel[grid](
        A_fp16, A_scale,
        B_fp8, B_scale,
        out,
        expert_offsets, expert_ids,
        weights,
        N, K,
        A_fp16.stride(0),
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


def _fused_routing(routing_logits, routing_bias, routed_scaling_factor):
    """Fused DeepSeek-V3 routing via single Triton kernel.

    Returns topk_idx [T, TOP_K] and topk_weights [T, TOP_K] (sparse).
    """
    T = routing_logits.shape[0]
    device = routing_logits.device
    group_size = E_GLOBAL // N_GROUP

    logits = routing_logits.to(torch.float32).contiguous()
    bias = routing_bias.to(torch.float32).reshape(-1).contiguous()

    topk_idx = torch.empty((T, TOP_K), dtype=torch.int64, device=device)
    topk_weights = torch.empty((T, TOP_K), dtype=torch.float32, device=device)

    if T > 0:
        _routing_kernel[(T,)](
            logits, bias,
            topk_idx, topk_weights,
            routed_scaling_factor,
            logits.stride(0),
            E_GLOBAL, N_GROUP, group_size, TOPK_GROUP, TOP_K,
            num_warps=4,
        )

    return topk_idx, topk_weights


def _pytorch_dispatch(topk_idx, topk_weights, local_start, E_local, T, device):
    """PyTorch-based dispatch for large T (V009 logic)."""
    token_ids = torch.arange(T, device=device).unsqueeze(1).expand(T, TOP_K).reshape(-1)
    expert_ids = topk_idx.reshape(-1)
    flat_weights = topk_weights.reshape(-1)
    local_expert_ids = expert_ids - local_start

    valid_mask = (local_expert_ids >= 0) & (local_expert_ids < E_local)
    valid_token_ids = token_ids[valid_mask]
    valid_local_expert_ids = local_expert_ids[valid_mask]
    valid_flat_weights = flat_weights[valid_mask]

    if valid_token_ids.numel() == 0:
        return None, None, None, None, 0, 0

    sorted_indices = torch.argsort(valid_local_expert_ids, stable=True)
    sorted_token_ids = valid_token_ids[sorted_indices]
    sorted_local_eids = valid_local_expert_ids[sorted_indices]
    w_sorted = valid_flat_weights[sorted_indices]

    unique_experts, expert_counts = torch.unique_consecutive(sorted_local_eids, return_counts=True)
    expert_offsets = torch.zeros(unique_experts.shape[0] + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = expert_counts.cumsum(0)
    N_sorted = sorted_token_ids.shape[0]
    max_count = expert_counts.max().item()

    return sorted_token_ids, w_sorted, expert_offsets, unique_experts, max_count, N_sorted


def _fused_dispatch(topk_idx, topk_weights, local_start, E_local, T):
    """Fused dispatch: single Triton kernel replaces ~15 PyTorch ops.

    Dense layout: expert_offsets covers all E_local experts.
    Returns sorted_token_ids, sorted_weights, expert_offsets, expert_ids,
            max_count, N_valid.
    """
    device = topk_idx.device
    total = T * TOP_K

    sorted_token_ids = torch.empty(total, dtype=torch.int64, device=device)
    sorted_weights = torch.empty(total, dtype=torch.float32, device=device)
    expert_offsets = torch.empty(E_local + 1, dtype=torch.int64, device=device)
    meta = torch.empty(2, dtype=torch.int64, device=device)

    _dispatch_kernel[(1,)](
        topk_idx.reshape(-1), topk_weights.reshape(-1),
        local_start,
        sorted_token_ids, sorted_weights,
        expert_offsets,
        meta,
        total, TOP_K, E_local,
        num_warps=1,
    )

    # Single DtoH sync for metadata (replaces ~7 syncs)
    meta_cpu = meta.tolist()
    max_count, N_valid = int(meta_cpu[0]), int(meta_cpu[1])

    # Dense expert_ids = all local experts [0..E_local-1]
    expert_ids = torch.arange(E_local, dtype=torch.int64, device=device)

    return (
        sorted_token_ids[:N_valid],
        sorted_weights[:N_valid],
        expert_offsets,
        expert_ids,
        max_count, N_valid,
    )


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
    """Fused MoE kernel: grouped GEMM + fused routing + fused dispatch (V012).

    Key changes from V009:
    1. Fused dispatch: histogram+prefix-sum+scatter replaces ~15 PyTorch ops
    2. Single DtoH sync (meta.tolist()) replaces ~7 syncs
    """
    device = hidden_states.device
    T = routing_logits.shape[0]
    E_local = gemm1_weights.shape[0]
    H = hidden_states.shape[1]
    I = gemm2_weights.shape[2]
    gemm1_out_size = gemm1_weights.shape[1]

    local_start = local_expert_offset.item() if isinstance(local_expert_offset, torch.Tensor) else int(local_expert_offset)
    scaling = routed_scaling_factor.item() if isinstance(routed_scaling_factor, torch.Tensor) else float(routed_scaling_factor)

    # ── Part A: Routing (single Triton kernel) ──
    topk_idx, topk_weights = _fused_routing(
        routing_logits, routing_bias, scaling,
    )

    # ── Part B+C: Dispatch — Triton for small T, PyTorch for large T ──
    if T * TOP_K <= 2048:
        # Fused Triton dispatch (eliminates ~15 PyTorch ops)
        sorted_token_ids, w_sorted, expert_offsets, unique_experts, max_count, N_sorted = \
            _fused_dispatch(topk_idx, topk_weights, local_start, E_local, T)
    else:
        # PyTorch dispatch (better for large T where GEMM dominates)
        sorted_token_ids, w_sorted, expert_offsets, unique_experts, max_count, N_sorted = \
            _pytorch_dispatch(topk_idx, topk_weights, local_start, E_local, T, device)

    if N_sorted == 0:
        output.zero_()
        return

    # ── Part D: Gather FP8 hidden_states + scales ──
    A_fp8_sorted = hidden_states[sorted_token_ids]
    A_scale_sorted = hidden_states_scale.T[sorted_token_ids]

    # ── Part E: Grouped GEMM1 + SwiGLU + FP8 Quant + GEMM2 ──
    G1_buf = torch.empty((N_sorted, gemm1_out_size), dtype=torch.float32, device=device)
    C_buf = torch.empty((N_sorted, I), dtype=torch.float32, device=device)
    O_buf = torch.empty((N_sorted, H), dtype=torch.float32, device=device)

    _grouped_fp8_dual_dequant_gemm(
        A_fp8_sorted, A_scale_sorted,
        gemm1_weights, gemm1_weights_scale,
        expert_offsets, unique_experts,
        max_count, gemm1_out_size, H,
        out=G1_buf,
    )

    _swiglu(G1_buf, N_sorted, I, out=C_buf)

    # ── FP16 pre-normalization of SwiGLU output for GEMM2 ──
    K2_sc = I // BLOCK  # = 2048 // 128 = 16
    c_blk = C_buf.reshape(N_sorted, K2_sc, BLOCK)
    c_scale = c_blk.abs().amax(dim=2).clamp(min=1e-8)   # [N_sorted, 16]
    C_fp16 = (c_blk / c_scale.unsqueeze(-1)).reshape(N_sorted, I).to(torch.float16).contiguous()

    _grouped_fp16_gemm2(
        C_fp16, c_scale,
        gemm2_weights, gemm2_weights_scale,
        expert_offsets, unique_experts,
        w_sorted, max_count, H, I,
        out=O_buf,
    )

    # ── Part F: Scatter ──
    result = torch.zeros((T, H), dtype=torch.float32, device=device)
    result.index_add_(0, sorted_token_ids, O_buf)

    output.copy_(result.to(torch.bfloat16))
