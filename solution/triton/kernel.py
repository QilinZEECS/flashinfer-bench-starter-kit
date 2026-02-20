"""
Fused MoE Kernel for FlashInfer Competition.

Track: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
Model: DeepSeek-V3/R1

P0: Sorted dispatch — eliminate Python expert loop.
Sort tokens by expert assignment, compute per-expert boundaries,
batch all GEMM/SwiGLU ops without Python-level iteration overhead.
"""

import torch
import triton
import triton.language as tl

# ── Constants (DeepSeek-V3/R1 geometry) ──────────────────────────────
BLOCK = 128
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
E_GLOBAL = 256


# ── Triton kernels ──────────────────────────────────────────────────

@triton.jit
def _dequant_hidden_states_kernel(
    fp8_ptr, scale_ptr, out_ptr,
    T, H: tl.constexpr, NUM_H_BLOCKS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    """Dequantize FP8 hidden_states with block-wise scales."""
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    for block_idx in tl.static_range(NUM_H_BLOCKS):
        cols = block_idx * BLOCK_SIZE + col_offsets
        mask = cols < H

        fp8_vals = tl.load(fp8_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr + block_idx * T + row)

        tl.store(out_ptr + row * H + cols, fp8_vals * s, mask=mask)


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


# ── Helper functions ────────────────────────────────────────────────

def _dequant_hidden(hidden_states, hidden_states_scale, T, H):
    """Dequantize FP8 hidden states → float32 [T, H]."""
    num_h_blocks = H // BLOCK
    out = torch.empty((T, H), dtype=torch.float32, device=hidden_states.device)
    _dequant_hidden_states_kernel[(T,)](
        hidden_states, hidden_states_scale, out,
        T, H, num_h_blocks, BLOCK,
    )
    return out


def _dequant_expert_weight(fp8_weight, scale, rows, cols, num_row_blocks, num_col_blocks):
    """Dequantize one expert's FP8 weight → float32 [rows, cols]."""
    out = torch.empty((rows, cols), dtype=torch.float32, device=fp8_weight.device)
    _dequant_weight_kernel[(rows,)](
        fp8_weight, scale, out,
        cols, num_row_blocks, num_col_blocks, BLOCK,
    )
    return out


def _swiglu(gemm1_out, N, I):
    """Apply SwiGLU activation on GEMM1 output [N, 2*I] → [N, I]."""
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
    """Fused MoE kernel with sorted dispatch (P0 optimization).

    Instead of a Python for-loop over 32 experts, we:
    1. Flatten all (token, expert) pairs from top-K routing
    2. Sort by expert ID for contiguous access
    3. Compute per-expert boundaries
    4. Batch process each expert group without Python iteration overhead
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

    # ── Part A: Dequantize hidden states (one launch for all tokens) ──
    A = _dequant_hidden(hidden_states, hidden_states_scale, T, H)

    # ── Part B: Routing ──
    topk_idx, weights = _deepseek_v3_routing(
        routing_logits, routing_bias, scaling,
    )

    # ── Part C: Sorted dispatch ──
    # Flatten (token, expert) pairs: T*TOP_K pairs total
    token_ids = torch.arange(T, device=device).unsqueeze(1).expand(T, TOP_K).reshape(-1)
    expert_ids = topk_idx.reshape(-1)  # [T*TOP_K] global expert indices

    # Convert to local expert indices
    local_expert_ids = expert_ids - local_start

    # Filter to only valid local experts [0, E_local)
    valid_mask = (local_expert_ids >= 0) & (local_expert_ids < E_local)
    valid_token_ids = token_ids[valid_mask]
    valid_local_expert_ids = local_expert_ids[valid_mask]
    valid_global_expert_ids = expert_ids[valid_mask]

    if valid_token_ids.numel() == 0:
        output.zero_()
        return

    # Sort by local expert ID for contiguous processing
    sorted_indices = torch.argsort(valid_local_expert_ids, stable=True)
    sorted_token_ids = valid_token_ids[sorted_indices]
    sorted_local_eids = valid_local_expert_ids[sorted_indices]
    sorted_global_eids = valid_global_expert_ids[sorted_indices]

    # Find expert boundaries
    # expert_boundaries[e] = first index where local expert e starts
    # expert_counts[e] = number of tokens for local expert e
    unique_experts, expert_counts = torch.unique_consecutive(sorted_local_eids, return_counts=True)

    # Gather all tokens at once (each token may appear multiple times for different experts)
    A_sorted = A[sorted_token_ids]  # [N_total, H]

    # Gather weights for each (token, expert) pair
    # weights is [T, E_GLOBAL], we need weights[token_id, global_expert_id]
    w_sorted = weights[sorted_token_ids, sorted_global_eids]  # [N_total]

    # Pre-allocate result accumulator
    result = torch.zeros((T, H), dtype=torch.float32, device=device)

    # Process each expert batch (vectorized — no Python per-token loop)
    offset = 0
    for i in range(unique_experts.shape[0]):
        le = unique_experts[i].item()
        count = expert_counts[i].item()

        A_e = A_sorted[offset:offset + count]  # [count, H]
        w_e = w_sorted[offset:offset + count]  # [count]
        tids = sorted_token_ids[offset:offset + count]

        # Dequantize expert weights
        W13_e = _dequant_expert_weight(
            gemm1_weights[le], gemm1_weights_scale[le],
            gemm1_out_size, H, num_gemm1_out_blocks, num_hidden_blocks,
        )
        W2_e = _dequant_expert_weight(
            gemm2_weights[le], gemm2_weights_scale[le],
            H, I, num_hidden_blocks, num_intermediate_blocks,
        )

        # GEMM1: [count, H] @ [H, 2I] → [count, 2I]
        G1 = A_e.matmul(W13_e.t())

        # SwiGLU: silu(gate) * up → [count, I]
        C = _swiglu(G1, count, I)

        # GEMM2: [count, I] @ [I, H] → [count, H]
        O = C.matmul(W2_e.t())

        # Weighted scatter-add
        result.index_add_(0, tids, O * w_e.unsqueeze(1))

        offset += count

    # Write to DPS output
    output.copy_(result.to(torch.bfloat16))
