"""
Fused MoE Kernel for FlashInfer Competition.

Track: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
Model: DeepSeek-V3/R1

V005: Fused dequant-GEMM (FP32/TF32).
- Fused dequant + GEMM: load FP8 weights, dequant to FP32 in registers, GEMM via tl.dot
- Eliminates intermediate weight buffers and separate dequant kernel launches
- FP32 dot (TF32 tensor cores) with FP32 accumulation
- Sorted dispatch for contiguous token processing
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

@triton.jit
def _dequant_hidden_kernel(
    fp8_ptr, scale_ptr, out_ptr,
    T, H: tl.constexpr, NUM_H_BLOCKS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    """Dequantize FP8 hidden_states -> float32."""
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    for block_idx in tl.static_range(NUM_H_BLOCKS):
        cols = block_idx * BLOCK_SIZE + col_offsets
        mask = cols < H

        fp8_vals = tl.load(fp8_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr + block_idx * T + row)

        tl.store(out_ptr + row * H + cols, fp8_vals * s, mask=mask)


@triton.jit
def _fused_dequant_gemm_kernel(
    # Pointers
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
    # Tile size
    BLOCK_M: tl.constexpr,
):
    """Fused dequant + GEMM: C[M,N] = A[M,K](f32) @ B[N,K](fp8).T

    B is FP8 with [N//128, K//128] block scale.
    BLOCK_N = BLOCK_K = 128 to match FP8 block scale granularity.
    Loads B^T as [BK, BN] directly (no tl.trans).
    FP32 dot (TF32 tensor cores), FP32 accumulation.
    """
    BLOCK_N: tl.constexpr = 128
    BLOCK_K: tl.constexpr = 128

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    # A pointers: load [BLOCK_M, BLOCK_K] tiles
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :]

    # B^T pointers: load [BLOCK_K, BLOCK_N] directly (B is [N,K] row-major)
    # B^T[k, n] = B[n, k] = B_ptr + n * stride_bn + k
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bt_ptrs = B_ptr + offs_n[None, :] * stride_bn + tl.arange(0, BLOCK_K)[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M

    for k_step in tl.static_range(K // BLOCK_K):
        # Load A tile [BLOCK_M, BLOCK_K] in FP32
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)

        # Load B^T tile [BLOCK_K, BLOCK_N] in FP8, dequant to FP32
        bt_fp8 = tl.load(bt_ptrs)
        scale = tl.load(B_scale_ptr + pid_n * num_scale_cols + k_step)
        bt_f32 = bt_fp8.to(tl.float32) * scale

        # GEMM: [BM, BK] @ [BK, BN] -> [BM, BN], FP32 dot (TF32 tensor cores)
        acc += tl.dot(a, bt_f32)

        # Advance K pointers
        a_ptrs += BLOCK_K
        bt_ptrs += BLOCK_K

    # Store result as FP32
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

def _dequant_hidden_f32(hidden_states, hidden_states_scale, T, H):
    """Dequantize FP8 hidden states -> float32 [T, H]."""
    num_h_blocks = H // BLOCK
    out = torch.empty((T, H), dtype=torch.float32, device=hidden_states.device)
    _dequant_hidden_kernel[(T,)](
        hidden_states, hidden_states_scale, out,
        T, H, num_h_blocks, BLOCK,
    )
    return out


def _fused_dequant_gemm(A, B_fp8, B_scale, M, N, K, BLOCK_M=64):
    """Fused dequant + GEMM: C = A(f32) @ B(fp8).T with block-scale dequant."""
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    num_scale_cols = K // BLOCK
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK))
    _fused_dequant_gemm_kernel[grid](
        A, B_fp8, B_scale, C,
        M, N, K,
        A.stride(0), B_fp8.stride(0),
        num_scale_cols,
        BLOCK_M,
    )
    return C


def _swiglu(gemm1_out, N, I):
    """SwiGLU on FP32 input -> FP32 output [N, I]."""
    out = torch.empty((N, I), dtype=torch.float32, device=gemm1_out.device)
    block_size = min(BLOCK, I)
    _swiglu_kernel[(N,)](gemm1_out, out, N, I, block_size)
    return out


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
    """Fused MoE kernel: sorted dispatch + fused dequant-GEMM (V005).

    Key optimizations over V001:
    1. Fused dequant + GEMM: FP8 weight dequant in GEMM tile loop
       - No intermediate FP32 weight buffers
       - No separate weight dequant kernel launches
    2. FP32 dot (TF32 tensor cores) with FP32 accumulation
    """
    device = hidden_states.device
    T = routing_logits.shape[0]
    E_local = gemm1_weights.shape[0]
    H = hidden_states.shape[1]
    I = gemm2_weights.shape[2]
    gemm1_out_size = gemm1_weights.shape[1]

    local_start = local_expert_offset.item() if isinstance(local_expert_offset, torch.Tensor) else int(local_expert_offset)
    scaling = routed_scaling_factor.item() if isinstance(routed_scaling_factor, torch.Tensor) else float(routed_scaling_factor)

    # ── Part A: Dequantize hidden states to FP32 (once for all experts) ──
    A = _dequant_hidden_f32(hidden_states, hidden_states_scale, T, H)

    # ── Part B: Routing ──
    topk_idx, weights = _deepseek_v3_routing(
        routing_logits, routing_bias, scaling,
    )

    # ── Part C: Sorted dispatch ──
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

    # Pre-gather inputs
    A_sorted = A[sorted_token_ids]  # [N_total, H] f32
    w_sorted = weights[sorted_token_ids, sorted_global_eids]

    result = torch.zeros((T, H), dtype=torch.float32, device=device)

    # ── Process each expert with fused dequant-GEMM ──
    offset = 0
    for i in range(unique_experts.shape[0]):
        le = unique_experts[i].item()
        count = expert_counts[i].item()

        tids = sorted_token_ids[offset:offset + count]
        w_e = w_sorted[offset:offset + count]
        A_e = A_sorted[offset:offset + count]  # [count, H] f32

        # Fused dequant + GEMM1: [count, H] @ [2I, H](fp8).T -> [count, 2I] f32
        G1 = _fused_dequant_gemm(
            A_e, gemm1_weights[le], gemm1_weights_scale[le],
            count, gemm1_out_size, H,
        )

        # SwiGLU: f32 -> f32
        C = _swiglu(G1, count, I)

        # Fused dequant + GEMM2: [count, I] @ [H, I](fp8).T -> [count, H] f32
        O = _fused_dequant_gemm(
            C, gemm2_weights[le], gemm2_weights_scale[le],
            count, H, I,
        )

        # Weighted accumulation in float32
        result.index_add_(0, tids, O * w_e.unsqueeze(1))

        offset += count

    output.copy_(result.to(torch.bfloat16))
