"""
Fused MoE Kernel — Triton grouped GEMM with FP8 block-scale dequant.

Strategy:
  - DeepSeek V3 routing in Python
  - Sort tokens by expert, build per-tile dispatch table
  - GEMM1: FP8 A (with block scale) × FP8 B (with block scale) → FP32, all tiles parallel
  - SwiGLU elementwise (PyTorch)
  - GEMM2: FP32 A → dynamic FP8 quant per tile × FP8 B (with block scale) → FP32, FP8 tensor cores
  - Vectorized weighted index_add_ (no per-expert Python loop)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

TOP_K      = 8
N_GROUP    = 8
TOPK_GROUP = 4
E_GLOBAL   = 256
BLOCK_SC   = 128   # FP8 block-scale size (fixed by weight format)


# ─── Triton GEMM1: FP8 A (with block scale) × FP8 B (with block scale) ───────

@triton.jit
def _gemm1_kernel(
    A_ptr, A_sc_ptr,           # A: [tv, K] FP8;  A_sc: [tv, K_sc] FP32
    B_ptr, B_sc_ptr,           # B: [E_loc, N, K] FP8;  B_sc: [E_loc, N_sc, K_sc]
    C_ptr,                     # C: [tv, N] FP32
    tile_expert_ptr, tile_row_ptr,
    exp_off_ptr, exp_cnt_ptr, exp_eid_ptr,
    N, K, K_sc, N_sc,
    stride_am, stride_ak,
    stride_ascm, stride_asck,
    stride_be, stride_bn, stride_bk,
    stride_bscE, stride_bscN, stride_bscK,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    expert_slot = tl.load(tile_expert_ptr + pid_m)
    m_block     = tl.load(tile_row_ptr    + pid_m)

    exp_off = tl.load(exp_off_ptr + expert_slot)
    exp_cnt = tl.load(exp_cnt_ptr + expert_slot)
    eid     = tl.load(exp_eid_ptr + expert_slot)

    m_start = m_block * BLOCK_M
    n_start = pid_n   * BLOCK_N

    m_offs      = m_start + tl.arange(0, BLOCK_M)
    n_offs      = n_start + tl.arange(0, BLOCK_N)
    k_offs_base = tl.arange(0, BLOCK_K)

    m_valid = m_offs < exp_cnt
    n_valid = n_offs < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    B_base   = B_ptr    + eid * stride_be
    Bsc_base = B_sc_ptr + eid * stride_bscE
    A_base   = A_ptr    + exp_off * stride_am
    Asc_base = A_sc_ptr + exp_off * stride_ascm
    C_base   = C_ptr    + exp_off * stride_cm

    for k_blk in range(tl.cdiv(K, BLOCK_K)):
        k_offs  = k_blk * BLOCK_K + k_offs_base
        k_valid = k_offs < K

        # A tile [BLOCK_M, BLOCK_K]  FP8  (keep as FP8 for tensor-core dot)
        a_ptrs = A_base + m_offs[:, None] * stride_am + k_offs[None, :]
        a_tile = tl.load(a_ptrs,
                         mask=m_valid[:, None] & k_valid[None, :],
                         other=0.0)

        # A scales [BLOCK_M] for this K-block
        asc_ptrs = Asc_base + m_offs * stride_ascm + k_blk
        a_sc = tl.load(asc_ptrs, mask=m_valid, other=1.0)

        # B tile [BLOCK_N, BLOCK_K]  FP8  (keep as FP8)
        b_ptrs = B_base + n_offs[:, None] * stride_bn + k_offs[None, :] * stride_bk
        b_tile = tl.load(b_ptrs,
                         mask=n_valid[:, None] & k_valid[None, :],
                         other=0.0)

        # B scale scalar for (n_block, k_block)
        b_sc = tl.load(Bsc_base + pid_n * stride_bscN + k_blk * stride_bscK)

        # FP8 tensor-core dot, accumulate in FP32, then apply block scales
        # raw_dot[m,n] = sum_k a_fp8[m,k] * b_fp8[n,k]  (FP8 tensor cores on B200)
        raw_dot = tl.dot(a_tile, tl.trans(b_tile), out_dtype=tl.float32)
        acc    += raw_dot * a_sc[:, None] * b_sc

    c_ptrs = C_base + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_valid[:, None] & n_valid[None, :])


# ─── Triton GEMM2: pre-scaled FP16 A × FP8 B (with block scale) ──────────────
# A is pre-normalized outside kernel (once per K-block per row), avoids redundant
# per-row max reductions across all N tiles.

@triton.jit
def _gemm2_kernel(
    A_ptr, A_sc_ptr,           # A: [tv, K] FP16 (pre-normalized);  A_sc: [tv, K_sc] FP32
    B_ptr, B_sc_ptr,           # B: [E_loc, N, K] FP8;  B_sc: [E_loc, N_sc, K_sc]
    C_ptr,                     # C: [tv, N] FP32
    tile_expert_ptr, tile_row_ptr,
    exp_off_ptr, exp_cnt_ptr, exp_eid_ptr,
    N, K, K_sc, N_sc,
    stride_am, stride_ak,
    stride_ascm, stride_asck,
    stride_be, stride_bn, stride_bk,
    stride_bscE, stride_bscN, stride_bscK,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    expert_slot = tl.load(tile_expert_ptr + pid_m)
    m_block     = tl.load(tile_row_ptr    + pid_m)

    exp_off = tl.load(exp_off_ptr + expert_slot)
    exp_cnt = tl.load(exp_cnt_ptr + expert_slot)
    eid     = tl.load(exp_eid_ptr + expert_slot)

    m_start = m_block * BLOCK_M
    n_start = pid_n   * BLOCK_N

    m_offs      = m_start + tl.arange(0, BLOCK_M)
    n_offs      = n_start + tl.arange(0, BLOCK_N)
    k_offs_base = tl.arange(0, BLOCK_K)

    m_valid = m_offs < exp_cnt
    n_valid = n_offs < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    B_base   = B_ptr    + eid * stride_be
    Bsc_base = B_sc_ptr + eid * stride_bscE
    A_base   = A_ptr    + exp_off * stride_am
    Asc_base = A_sc_ptr + exp_off * stride_ascm
    C_base   = C_ptr    + exp_off * stride_cm

    for k_blk in range(tl.cdiv(K, BLOCK_K)):
        k_offs  = k_blk * BLOCK_K + k_offs_base
        k_valid = k_offs < K

        # A tile [BLOCK_M, BLOCK_K]  FP16 (pre-normalized, load as-is)
        a_ptrs = A_base + m_offs[:, None] * stride_am + k_offs[None, :]
        a_tile = tl.load(a_ptrs,
                         mask=m_valid[:, None] & k_valid[None, :],
                         other=0.0)

        # A_sc [BLOCK_M]: per-row scale for this K-block (pre-computed outside kernel)
        asc_ptrs = Asc_base + m_offs * stride_ascm + k_blk
        a_sc = tl.load(asc_ptrs, mask=m_valid, other=1.0)

        # B tile [BLOCK_N, BLOCK_K]  FP8 → FP16 for tensor-core dot
        b_ptrs = B_base + n_offs[:, None] * stride_bn + k_offs[None, :] * stride_bk
        b_tile = tl.load(b_ptrs,
                         mask=n_valid[:, None] & k_valid[None, :],
                         other=0.0).to(tl.float16)

        # B scale scalar
        b_sc = tl.load(Bsc_base + pid_n * stride_bscN + k_blk * stride_bscK)

        # FP16×FP16→FP32 tensor-core dot, apply A per-row scale + B scale after
        raw_dot = tl.dot(a_tile, tl.trans(b_tile), out_dtype=tl.float32)
        acc    += raw_dot * a_sc[:, None] * b_sc

    c_ptrs = C_base + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_valid[:, None] & n_valid[None, :])


# ─── Python dispatch helpers ─────────────────────────────────────────────────

def _build_tile_map(cnt_list, BLOCK_M, device):
    expert_slots, tile_rows = [], []
    for slot, cnt in enumerate(cnt_list):
        n_tiles = (cnt + BLOCK_M - 1) // BLOCK_M
        expert_slots.extend([slot] * n_tiles)
        tile_rows.extend(range(n_tiles))
    return (
        torch.tensor(expert_slots, dtype=torch.int32, device=device),
        torch.tensor(tile_rows,    dtype=torch.int32, device=device),
    )


def _run_gemm1(A, A_sc, B, B_sc, C, tile_expert, tile_row,
               exp_off, exp_cnt, exp_eid, BLOCK_M=64, BLOCK_N=128):
    tv, K = A.shape
    N = B.shape[1]
    K_sc = A_sc.shape[1]
    N_sc = B_sc.shape[1]
    grid = (tile_expert.shape[0], triton.cdiv(N, BLOCK_N))
    _gemm1_kernel[grid](
        A, A_sc, B, B_sc, C,
        tile_expert, tile_row, exp_off, exp_cnt, exp_eid,
        N, K, K_sc, N_sc,
        A.stride(0), A.stride(1),
        A_sc.stride(0), A_sc.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        B_sc.stride(0), B_sc.stride(1), B_sc.stride(2),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_SC,
    )


def _run_gemm2(A_fp16, A_sc, B, B_sc, C, tile_expert, tile_row,
               exp_off, exp_cnt, exp_eid, BLOCK_M=64, BLOCK_N=128):
    """A_fp16: [tv, K] FP16 (pre-normalized); A_sc: [tv, K_sc] FP32 (per-row-kblock scales)."""
    tv, K = A_fp16.shape
    N = B.shape[1]
    K_sc = A_sc.shape[1]
    N_sc = B_sc.shape[1]
    grid = (tile_expert.shape[0], triton.cdiv(N, BLOCK_N))
    _gemm2_kernel[grid](
        A_fp16, A_sc, B, B_sc, C,
        tile_expert, tile_row, exp_off, exp_cnt, exp_eid,
        N, K, K_sc, N_sc,
        A_fp16.stride(0), A_fp16.stride(1),
        A_sc.stride(0), A_sc.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        B_sc.stride(0), B_sc.stride(1), B_sc.stride(2),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_SC,
    )


# ─── Routing ─────────────────────────────────────────────────────────────────

def _routing(routing_logits, routing_bias, routed_scaling_factor):
    T      = routing_logits.shape[0]
    device = routing_logits.device

    logits = routing_logits.float()
    bias   = routing_bias.float().reshape(-1)
    s           = torch.sigmoid(logits)
    s_with_bias = s + bias

    group_size   = E_GLOBAL // N_GROUP
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
    top2_vals    = s_wb_grouped.topk(2, dim=2).values
    group_scores = top2_vals.sum(dim=2)

    group_idx  = group_scores.topk(TOPK_GROUP, dim=1).indices
    group_mask = torch.zeros(T, N_GROUP, device=device)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_GLOBAL)

    neg_inf       = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    topk_idx      = scores_pruned.topk(TOP_K, dim=1).indices

    topk_scores  = s.gather(1, topk_idx)
    weights_sum  = topk_scores.sum(dim=1, keepdim=True) + 1e-20
    topk_weights = (topk_scores / weights_sum) * routed_scaling_factor

    return topk_idx.contiguous(), topk_weights.contiguous()


# ─── Main kernel ─────────────────────────────────────────────────────────────

def kernel(
    routing_logits,
    routing_bias,
    hidden_states,         # [T, H=7168]        float8_e4m3fn
    hidden_states_scale,   # [56, T]            float32 (transposed)
    gemm1_weights,         # [E, 2I=4096, H]    float8_e4m3fn
    gemm1_weights_scale,   # [E, 32, 56]        float32
    gemm2_weights,         # [E, H, I=2048]     float8_e4m3fn
    gemm2_weights_scale,   # [E, 56, 16]        float32
    local_expert_offset,
    routed_scaling_factor,
    output,
):
    T      = routing_logits.shape[0]
    E_loc  = gemm1_weights.shape[0]
    H      = hidden_states.shape[1]
    I      = gemm2_weights.shape[2]
    device = hidden_states.device

    local_start = (local_expert_offset.item()
                   if isinstance(local_expert_offset, torch.Tensor)
                   else int(local_expert_offset))
    scaling = (routed_scaling_factor.item()
               if isinstance(routed_scaling_factor, torch.Tensor)
               else float(routed_scaling_factor))

    # ── Routing ──────────────────────────────────────────────────────────────
    topk_idx, topk_weights = _routing(routing_logits, routing_bias, scaling)

    token_ids    = torch.arange(T, device=device).unsqueeze(1).expand(T, TOP_K).reshape(-1)
    expert_ids   = topk_idx.reshape(-1)
    flat_weights = topk_weights.reshape(-1)
    local_eids   = expert_ids - local_start

    valid = (local_eids >= 0) & (local_eids < E_loc)
    v_tok = token_ids[valid]
    v_eid = local_eids[valid]
    v_wt  = flat_weights[valid]

    if v_tok.numel() == 0:
        output.zero_()
        return

    order          = v_eid.argsort(stable=True)
    sorted_tids    = v_tok[order]
    sorted_eids    = v_eid[order]
    sorted_weights = v_wt[order]

    unique_exp, exp_counts = torch.unique_consecutive(sorted_eids, return_counts=True)
    exp_list = unique_exp.tolist()
    cnt_list = exp_counts.tolist()
    n_active = len(exp_list)
    tv       = sorted_tids.numel()

    # Per-active-expert tensors
    exp_off_t = torch.zeros(n_active, dtype=torch.int32, device=device)
    if n_active > 1:
        exp_off_t[1:] = exp_counts[:-1].cumsum(0).int()
    exp_cnt_t = exp_counts.int()
    exp_eid_t = unique_exp.int()

    # ── Gather sorted activations + scales ───────────────────────────────────
    a_sorted    = hidden_states[sorted_tids].contiguous()       # [tv, H] FP8
    a_sc_sorted = hidden_states_scale.T[sorted_tids].contiguous()  # [tv, 56] FP32

    # ── GEMM1 via Triton ──────────────────────────────────────────────────────
    BLOCK_M = 64
    BLOCK_N = 128

    tile_exp1, tile_row1 = _build_tile_map(cnt_list, BLOCK_M, device)

    g1_out = torch.empty(tv, 2 * I, dtype=torch.float32, device=device)

    _run_gemm1(
        a_sorted, a_sc_sorted,
        gemm1_weights, gemm1_weights_scale,
        g1_out,
        tile_exp1, tile_row1, exp_off_t, exp_cnt_t, exp_eid_t,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )

    # ── SwiGLU: first I=up, last I=gate ──────────────────────────────────────
    c_out = F.silu(g1_out[:, I:]) * g1_out[:, :I]   # [tv, I] FP32

    # ── GEMM2 via Triton ──────────────────────────────────────────────────────
    # c_out [tv, I] @ W2[E, H, I].T → [tv, H]
    # Pre-compute per-row-kblock FP16 normalization of c_out so the kernel
    # can load pre-scaled FP16 A without redundant reductions across N tiles.
    K2_sc = I // BLOCK_SC   # = 16  (number of K-blocks in GEMM2)
    c_blk = c_out.reshape(tv, K2_sc, BLOCK_SC)          # [tv, 16, 128] FP32
    c_sc  = c_blk.abs().amax(dim=2).clamp(min=1e-8)     # [tv, 16] FP32
    c_fp16 = (c_blk / c_sc.unsqueeze(-1)).reshape(tv, I).to(torch.float16).contiguous()

    tile_exp2, tile_row2 = _build_tile_map(cnt_list, BLOCK_M, device)

    o_out = torch.empty(tv, H, dtype=torch.float32, device=device)

    _run_gemm2(
        c_fp16, c_sc,
        gemm2_weights, gemm2_weights_scale,
        o_out,
        tile_exp2, tile_row2, exp_off_t, exp_cnt_t, exp_eid_t,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )

    # ── Vectorized weighted accumulate (no per-expert loop) ──────────────────
    result = torch.zeros(T, H, dtype=torch.float32, device=device)
    result.index_add_(0, sorted_tids, o_out * sorted_weights.unsqueeze(1))

    output.copy_(result.to(torch.bfloat16))
