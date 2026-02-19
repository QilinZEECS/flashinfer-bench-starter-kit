"""
Fused MoE Kernel for FlashInfer Competition.

Track: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
Model: DeepSeek-V3/R1

Initial version: correctness-first implementation using PyTorch ops
with Triton JIT kernels for the compute-intensive dequantization path.
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
    """Dequantize FP8 hidden_states with block-wise scales.

    fp8:   [T, H]           float8_e4m3fn
    scale: [NUM_H_BLOCKS, T] float32   (transposed layout)
    out:   [T, H]           float32
    """
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
    """Dequantize one expert's FP8 weight matrix with block-wise scales.

    fp8:   [ROWS, COLS]                     float8_e4m3fn
    scale: [NUM_ROW_BLOCKS, NUM_COL_BLOCKS] float32
    out:   [ROWS, COLS]                     float32
    """
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
    """SwiGLU activation: silu(gate) * up.

    Reference mapping: X1 (up) = first I cols, X2 (gate) = last I cols
    Result: C = silu(X2) * X1

    input: [N, 2*I]  (first I cols = up/X1, last I cols = gate/X2)
    out:   [N, I]
    """
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
    """DeepSeek-V3 no-aux routing.

    Returns:
        topk_idx:  [T, TOP_K]  int64 - selected expert indices
        weights:   [T, E_GLOBAL] float32 - per-expert combination weights
    """
    T = routing_logits.shape[0]
    device = routing_logits.device

    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)

    # Sigmoid scores
    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    # Group scoring: 8 groups of 32 experts each
    group_size = E_GLOBAL // N_GROUP
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)

    # Top-2 per group → group scores
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    # Select top-4 groups
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros(T, N_GROUP, device=device)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask
        .unsqueeze(2)
        .expand(T, N_GROUP, group_size)
        .reshape(T, E_GLOBAL)
    )

    # Global top-8 within kept groups (use s_with_bias)
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    # Combination weights: use s (without bias), normalize, scale
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
    """Fused MoE kernel: FP8 block-scale, DeepSeek-V3 routing, SwiGLU.

    Args:
        routing_logits:        [T, 256]        float32
        routing_bias:          [256]            bfloat16
        hidden_states:         [T, 7168]       float8_e4m3fn
        hidden_states_scale:   [56, T]         float32
        gemm1_weights:         [32, 4096, 7168] float8_e4m3fn
        gemm1_weights_scale:   [32, 32, 56]    float32
        gemm2_weights:         [32, 7168, 2048] float8_e4m3fn
        gemm2_weights_scale:   [32, 56, 16]    float32
        local_expert_offset:   int             scalar
        routed_scaling_factor: float           scalar
        output:                [T, 7168]       bfloat16  (DPS destination)
    """
    device = hidden_states.device
    T = routing_logits.shape[0]
    E_local = gemm1_weights.shape[0]
    H = hidden_states.shape[1]
    I = gemm2_weights.shape[2]
    gemm1_out_size = gemm1_weights.shape[1]  # 2 * I

    num_hidden_blocks = H // BLOCK
    num_intermediate_blocks = I // BLOCK
    num_gemm1_out_blocks = gemm1_out_size // BLOCK

    local_start = local_expert_offset.item() if isinstance(local_expert_offset, torch.Tensor) else int(local_expert_offset)
    scaling = routed_scaling_factor.item() if isinstance(routed_scaling_factor, torch.Tensor) else float(routed_scaling_factor)

    # ── Part A: Dequantize hidden states ──
    A = _dequant_hidden(hidden_states, hidden_states_scale, T, H)

    # ── Part B: Routing ──
    topk_idx, weights = _deepseek_v3_routing(
        routing_logits, routing_bias, scaling,
    )

    # ── Part C: Per-expert compute & accumulation ──
    result = torch.zeros((T, H), dtype=torch.float32, device=device)

    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_GLOBAL:
            continue

        # Find tokens that selected this expert
        sel_mask = (topk_idx == ge).any(dim=1)
        if not sel_mask.any():
            continue

        token_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
        Tk = token_idx.numel()

        # Gather inputs for selected tokens
        A_e = A.index_select(0, token_idx)

        # Dequantize expert weights
        W13_e = _dequant_expert_weight(
            gemm1_weights[le], gemm1_weights_scale[le],
            gemm1_out_size, H, num_gemm1_out_blocks, num_hidden_blocks,
        )
        W2_e = _dequant_expert_weight(
            gemm2_weights[le], gemm2_weights_scale[le],
            H, I, num_hidden_blocks, num_intermediate_blocks,
        )

        # GEMM1: [Tk, H] @ [H, 2I] → [Tk, 2I]
        G1 = A_e.matmul(W13_e.t())

        # SwiGLU: silu(gate) * up → [Tk, I]
        C = _swiglu(G1, Tk, I)

        # GEMM2: [Tk, I] @ [I, H] → [Tk, H]
        O = C.matmul(W2_e.t())

        # Weighted accumulation
        w_tok = weights.index_select(0, token_idx)[:, ge].unsqueeze(1)
        result.index_add_(0, token_idx, O * w_tok)

    # Write to DPS output
    output.copy_(result.to(torch.bfloat16))
