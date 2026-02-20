"""
Fused MoE Kernel for FlashInfer Competition.

Track: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
Model: DeepSeek-V3/R1

V002: Sorted dispatch + FP8 scaled_mm.
- Sorted dispatch (from P0) eliminates Python per-token overhead
- torch._scaled_mm uses B200 FP8 tensor cores instead of dequant->FP32->cublas
- Row/col-wise scale approximation from block-wise scales
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


# ── Triton SwiGLU kernel ──────────────────────────────────────────

@triton.jit
def _swiglu_kernel(
    input_ptr, out_ptr,
    N, I: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    """SwiGLU activation: silu(gate) * up."""
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
def _dequant_weight_kernel(
    fp8_ptr, scale_ptr, out_ptr,
    COLS: tl.constexpr,
    NUM_ROW_BLOCKS: tl.constexpr, NUM_COL_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Dequantize one expert's FP8 weight matrix with block-wise scales."""
    row = tl.program_id(0)
    row_block = row // BLOCK_SIZE
    col_offsets = tl.arange(0, BLOCK_SIZE)

    for col_block in tl.static_range(NUM_COL_BLOCKS):
        cols = col_block * BLOCK_SIZE + col_offsets
        mask = cols < COLS

        fp8_vals = tl.load(fp8_ptr + row * COLS + cols, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr + row_block * NUM_COL_BLOCKS + col_block)

        tl.store(out_ptr + row * COLS + cols, fp8_vals * s, mask=mask)


# ── Helper functions ────────────────────────────────────────────────

def _swiglu(gemm1_out, N, I):
    """Apply SwiGLU activation on GEMM1 output [N, 2*I] -> [N, I]."""
    out = torch.empty((N, I), dtype=torch.float32, device=gemm1_out.device)
    block_size = min(BLOCK, I)
    _swiglu_kernel[(N,)](gemm1_out, out, N, I, block_size)
    return out


def _dequant_expert_weight(fp8_weight, scale, rows, cols, num_row_blocks, num_col_blocks):
    """Dequantize one expert's FP8 weight -> float32 [rows, cols]."""
    out = torch.empty((rows, cols), dtype=torch.float32, device=fp8_weight.device)
    _dequant_weight_kernel[(rows,)](
        fp8_weight, scale, out,
        cols, num_row_blocks, num_col_blocks, BLOCK,
    )
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


def _quantize_to_fp8(x):
    """Quantize float32 tensor to float8_e4m3fn with per-row scale."""
    max_per_row = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = max_per_row / 448.0
    x_scaled = x / scale
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    return x_fp8, scale


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
    """Fused MoE kernel with sorted dispatch + FP8 scaled_mm (V002).

    Key optimizations over baseline:
    1. Sorted dispatch: batch tokens per expert, contiguous access
    2. torch._scaled_mm for GEMM1: FP8 tensor cores, no dequantization
    3. Quantize intermediate to FP8 for GEMM2: both GEMMs use tensor cores
    """
    device = hidden_states.device
    T = routing_logits.shape[0]
    E_local = gemm1_weights.shape[0]
    H = hidden_states.shape[1]
    I = gemm2_weights.shape[2]
    gemm1_out_size = gemm1_weights.shape[1]

    num_hidden_blocks = H // BLOCK
    num_intermediate_blocks = I // BLOCK
    num_gemm1_out_blocks = gemm1_out_size // BLOCK

    local_start = local_expert_offset.item() if isinstance(local_expert_offset, torch.Tensor) else int(local_expert_offset)
    scaling = routed_scaling_factor.item() if isinstance(routed_scaling_factor, torch.Tensor) else float(routed_scaling_factor)

    # ── Routing ──
    topk_idx, weights = _deepseek_v3_routing(
        routing_logits, routing_bias, scaling,
    )

    # ── Sorted dispatch setup ──
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

    # ── Precompute transposed weights for _scaled_mm ──
    # _scaled_mm(A[M,K], B[K,N]) = C[M,N]
    W1_T = gemm1_weights.transpose(1, 2).contiguous()  # [E, H, 2I]
    W2_T = gemm2_weights.transpose(1, 2).contiguous()  # [E, I, H]

    # ── Precompute col-wise scale approximations ──
    # GEMM1 W scale: [E, 2I/128, H/128] → mean over K(H)-blocks → [E, 2I/128]
    W1_col_scale = gemm1_weights_scale.mean(dim=2)  # [E, 32]
    # GEMM2 W scale: [E, H/128, I/128] → mean over K(I)-blocks → [E, H/128]
    W2_col_scale = gemm2_weights_scale.mean(dim=2)  # [E, 56]

    # ── Pre-gather FP8 hidden states ──
    A_fp8_sorted = hidden_states[sorted_token_ids]  # [N_total, H] fp8
    w_sorted = weights[sorted_token_ids, sorted_global_eids]

    result = torch.zeros((T, H), dtype=torch.float32, device=device)

    # ── Process each expert ──
    offset = 0
    for i in range(unique_experts.shape[0]):
        le = unique_experts[i].item()
        count = expert_counts[i].item()

        tids = sorted_token_ids[offset:offset + count]
        w_e = w_sorted[offset:offset + count]

        # ── GEMM1: [count, H] @ [H, 2I] → [count, 2I] ──
        A_e_fp8 = A_fp8_sorted[offset:offset + count].contiguous()

        # Row-wise scale: mean over H-blocks per token
        a_row_scale = hidden_states_scale[:, tids].mean(dim=0).unsqueeze(1).to(torch.float32)

        # Col-wise scale: [32] → [1, 4096]
        w1_scale = W1_col_scale[le].repeat_interleave(BLOCK).unsqueeze(0).to(torch.float32)

        try:
            G1 = torch._scaled_mm(
                A_e_fp8,
                W1_T[le],
                scale_a=a_row_scale,
                scale_b=w1_scale,
                out_dtype=torch.float32,
            )
        except Exception:
            # Fallback to dequant + matmul
            A_e_f32 = A_e_fp8.to(torch.float32)
            for bk in range(num_hidden_blocks):
                s = hidden_states_scale[bk, tids].unsqueeze(1)
                A_e_f32[:, bk*BLOCK:(bk+1)*BLOCK] *= s
            W1_e = _dequant_expert_weight(
                gemm1_weights[le], gemm1_weights_scale[le],
                gemm1_out_size, H, num_gemm1_out_blocks, num_hidden_blocks,
            )
            G1 = A_e_f32.matmul(W1_e.t())

        # ── SwiGLU ──
        C = _swiglu(G1, count, I)

        # ── GEMM2: [count, I] @ [I, H] → [count, H] ──
        C_fp8, c_scale = _quantize_to_fp8(C)

        # Col-wise scale: [56] → [1, 7168]
        w2_scale = W2_col_scale[le].repeat_interleave(BLOCK).unsqueeze(0).to(torch.float32)

        try:
            O = torch._scaled_mm(
                C_fp8,
                W2_T[le],
                scale_a=c_scale.to(torch.float32),
                scale_b=w2_scale,
                out_dtype=torch.float32,
            )
        except Exception:
            # Fallback to dequant + matmul
            W2_e = _dequant_expert_weight(
                gemm2_weights[le], gemm2_weights_scale[le],
                H, I, num_hidden_blocks, num_intermediate_blocks,
            )
            O = C.matmul(W2_e.t())

        # ── Weighted scatter-add ──
        result.index_add_(0, tids, O * w_e.unsqueeze(1))

        offset += count

    output.copy_(result.to(torch.bfloat16))
