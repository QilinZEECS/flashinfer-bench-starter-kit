# GPU Kernel Optimization Guide

**Lessons from FlashInfer Fused MoE Optimization (V001→V009, 1.2x→15.6x on B200)**

---

## 1. Methodology: Get It Right, Then Make It Fast

### 1.1 Iterative Rhythm

```
V001: Correct baseline              ← correctness first
V005: Eliminate obvious memory waste ← low-hanging fruit
V006: Micro-tune (autotune, batch)  ← reduce overhead
V007: Use faster hardware ops       ← exploit hardware features
V008: Change parallelism model      ← architectural rethink
V009: Operator fusion               ← squeeze the last drops
```

**Golden rule**: Change one thing per version, benchmark, confirm the effect, then move on. Changing two things at once makes bugs impossible to locate.

### 1.2 When to Stop

- Profiler shows the bottleneck shifted from compute to memory bandwidth → approaching hardware ceiling
- Optimization yields < 5% → diminishing returns, change direction
- Change has no effect or regresses → assumption was wrong, revert and analyze

---

## 2. Precision: Work Backwards from Hardware Instructions

### 2.1 Precision Hierarchy

```
FP8 e4m3  (3-bit mantissa)  → 9000 TFLOPS (B200)  → storage & transport
TF32      (10-bit mantissa) → 2250 TFLOPS          → default GEMM choice
FP32      (23-bit mantissa) → 113 TFLOPS            → accumulators & scalar math
```

### 2.2 Core Principle

**Use low precision for storage, but always accumulate in high precision.**

```python
# Correct: FP8 storage + FP32 accumulation
raw = tl.dot(a_fp8, bt_fp8)              # FP8 tensor core, FP32 accumulator
acc += raw * scale_a[:, None] * scale_b   # FP32 scale correction

# Wrong: BF16 accumulation
acc += tl.dot(a_bf16, b_bf16)  # BF16 dot accumulates with 7-bit mantissa
                                # over K=2048 iterations → error explodes
```

### 2.3 Precision Decision Tree

```
Input is FP8?
  ├── Both A and B are FP8 → Use FP8 tensor cores (tl.dot eats FP8 directly)
  │                           Apply scales AFTER dot (requires BLOCK_K == quant block size)
  │
  └── Only B is FP8 → Dequant B to FP32, then use TF32 dot
                       Cannot dequant to BF16/FP16 (precision loss or overflow)

Intermediate precision?
  ├── SwiGLU output → Must be FP32 (BF16 as GEMM2 A input → 11/19 workloads fail)
  └── Scatter accumulation → Must be FP32 (BF16 index_add_ fails at precision boundary)
```

### 2.4 FP8 Block-Scale Dequant: The Key Trick

```python
# When BLOCK_K == quantization block size (128), scales can be factored out of dot:
raw = tl.dot(a_fp8, bt_fp8)                  # Pure FP8 numerical product
acc += raw * scale_a[:, None] * scale_b       # Element-wise FP32 multiply

# Why this works:
#   real[i,j] = Σ_k (a[i,k]*sa[i]) * (b[k,j]*sb[j])
#             = sa[i] * sb * Σ_k a[i,k]*b[k,j]
#             = sa[i] * sb * raw[i,j]
# All k values in one k_step belong to the same scale block.
```

If BLOCK_K ≠ quantization block size, this simplification breaks. You must handle per-block scaling within the tile.

---

## 3. Triton Pitfalls

### 3.1 tl.trans() Is Broken on Some GPUs

```python
# Dangerous: may produce wrong GEMM results on B200
bt = tl.trans(tl.load(B_ptr))

# Safe: load transposed layout via pointer arithmetic
bt = tl.load(B_ptr + n[None, :] * stride_bn + k[:, None])
```

**Lesson**: When GEMM results are wrong, suspect `tl.trans()` first. Verify with small matrices and known answers.

### 3.2 Autotune: The `key` Parameter Controls Compilation Explosion

```python
# Bad: key=['M','N','K'] — recompiles for every unique M, M varies per workload
@triton.autotune(configs=configs, key=['M', 'N', 'K'])

# Good: key=['N','K'] — only 2 (N,K) shapes × 4 configs = 8 compilations
@triton.autotune(configs=configs, key=['N', 'K'])
```

In Grouped GEMM, `max_count` (the M dimension) changes every call. Including it in `key` causes every call to miss the autotuning cache.

### 3.3 tl.static_range vs range

```python
# Compile-time unrolled, iteration count must be constexpr
for k_step in tl.static_range(NUM_K_BLOCKS):  # NUM_K_BLOCKS: tl.constexpr
    ...

# Runtime loop, more flexible but potentially slower
for k_step in range(num_k_blocks):
    ...
```

Use `tl.static_range` for GEMM K-loops — the compiler can apply software pipelining.

### 3.4 Masking Has a Performance Cost

```python
# M dimension needs masking (expert token count may not be a multiple of BLOCK_M)
a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)

# N dimension: if alignment is guaranteed, skip the mask
bt = tl.load(bt_ptrs)  # N=7168, BLOCK_N=128, 7168/128=56, divides evenly
```

**Avoid masks where possible** — they introduce extra predicate registers and conditional execution.

### 3.5 Tuning num_warps and num_stages

```python
triton.Config({'BLOCK_M': 32},  num_warps=4, num_stages=3)   # Small tile, fewer warps
triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=4)   # Large tile, more warps

# num_warps: affects SM occupancy and register pressure
#   Large BLOCK_M → needs more warps to fill the compute pipeline
#   Small BLOCK_M → too many warps causes register spilling
#
# num_stages: software pipelining depth
#   More stages → better load/compute overlap, but uses more shared memory
#   B200 has plenty of shared memory; try 4-5 stages
```

---

## 4. Parallelism: From Python Loops to Grouped GEMM

### 4.1 Evolution Path

```
Level 0: Python for-loop + per-expert kernel launch
         → 32 experts × 3 kernels = 96 launches, massive Python overhead

Level 1: Batch dispatch (reduce Python synchronization)
         → Still 96 launches, but replace .item() with batch .tolist()

Level 2: Grouped GEMM with 3D grid (eliminate Python loop entirely)
         → 3 launches total (GEMM1 + SwiGLU + GEMM2)
         → Small workloads jumped from 8x to 12x
```

### 4.2 3D Grid Design

```python
grid = (num_experts, max_m_tiles, N // BLOCK_N)

# program_id(0) = expert index
# program_id(1) = M tile index
# program_id(2) = N tile index

# Key: early exit for uneven expert sizes
eidx = tl.program_id(0)
row_start = tl.load(expert_offsets_ptr + eidx)
row_end = tl.load(expert_offsets_ptr + eidx + 1)
count = row_end - row_start

if pid_m * BLOCK_M >= count:
    return  # This expert doesn't have enough tokens for this M tile
```

**Downside**: `max_m_tiles` is sized to the largest expert. Small experts waste many idle CTAs.

### 4.3 Advanced: Persistent Kernels (Not Yet Implemented)

```python
# Fixed CTA count = number of SMs, each CTA pulls work from a queue
pid = tl.program_id(0)  # 0..191 (B200 has 192 SMs)
while True:
    work_id = tl.atomic_add(global_counter, 1)
    if work_id >= total_work:
        return
    expert, m_tile, n_tile = decode(work_id)
    # ... process tile ...
```

Eliminates idle CTA waste, but significantly more complex to implement.

---

## 5. Operator Fusion: What's Worth Fusing

### 5.1 Decision Framework

```
Worth fusing:
  ✓ Eliminates large intermediate buffers (e.g., 84MB weight dequant buffer)
  ✓ Eliminates kernel launches (especially inside Python loops)
  ✓ Eliminates GPU↔CPU synchronization (.item(), .tolist())
  ✓ Lightweight ops folded into a heavy kernel's prologue/epilogue

Not worth fusing:
  ✗ Fusion increases register pressure enough to reduce occupancy
  ✗ The fused kernel accounts for < 1% of total time
  ✗ Added complexity with no benchmark improvement
```

### 5.2 Case Studies

```
✓ Weight dequant → GEMM tile loop (V005)
  Before: 84MB temp buffer + separate kernel
  After:  Dequant in registers, zero extra memory
  Result: Large workloads 1.2x → 2.1x

✓ Routing → single Triton kernel (V009)
  Before: ~10 PyTorch CUDA dispatch ops
  After:  1 kernel, sparse output
  Result: Reduced launch overhead + eliminated dense [T, 256] allocation

✓ Weight multiply → GEMM2 store epilogue (V009)
  Multiply by weight before tl.store, near-zero overhead
  Eliminated: temp buffer + extra kernel

✗ SwiGLU → GEMM2 prologue (V011)
  Read up/gate columns from G1_buf inline in GEMM2 tile loop
  Result: On par with V009 — register pressure increase offset bandwidth savings
  Lesson: SwiGLU itself is too lightweight (~1% of total time), fusion gain ≈ 0
```

### 5.3 Epilogue Fusion Pattern

The safest fusion: add element-wise operations during the GEMM store phase.

```python
# GEMM result is in the accumulator register — it must be written to HBM anyway
# Adding element-wise work before the store is nearly free:

# Example: fused weight multiply
w = tl.load(weights_ptr + offs_m, mask=mask_m, other=0.0)
acc = acc * w[:, None]           # FP32 × FP32, done in registers
tl.store(c_ptrs, acc, mask=...)  # Write to HBM

# Good candidates for epilogue fusion: scale, bias, activation (ReLU, clamp), type cast
# Bad candidates: cross-row/cross-column reductions (softmax, layernorm)
```

---

## 6. Performance Analysis: Memory-Bound vs Compute-Bound

### 6.1 Arithmetic Intensity

```
Arithmetic Intensity (AI) = FLOPs / Bytes

GEMM (M, N, K):
  FLOPs = 2 × M × N × K
  Bytes = (M×K + K×N + M×N) × sizeof(dtype)
  AI = 2MNK / ((MK + KN + MN) × dtype_size)
```

### 6.2 Roofline Analysis

```
B200 specs:
  FP8 tensor core:  9000 TFLOPS
  TF32 tensor core: 2250 TFLOPS
  HBM bandwidth:    8 TB/s

Balance point = Peak FLOPS / Bandwidth
  FP8:  9000 / 8 = 1125 FLOPs/byte  → need AI > 1125 to be compute-bound
  TF32: 2250 / 8 = 281 FLOPs/byte   → need AI > 281 to be compute-bound

Typical MoE GEMM2 (M=256, K=2048, N=7168):
  AI ≈ 110 FLOPs/byte → memory-bound (well below 281)

Conclusion: For large workloads, the GEMM bottleneck is loading weights from HBM,
            not computation. 4x compute speedup from FP8 tensor cores ≠ 4x end-to-end.
```

### 6.3 Optimizing Memory-Bound GEMMs

When GEMM is already memory-bound:
```
1. Reduce data movement (FP8 vs FP32 storage)
2. Improve L2 cache hit rate (tile ordering)
3. Persistent kernels (avoid redundant global memory reads)
4. Fuse surrounding operators (eliminate extra read/write of intermediate buffers)
5. Larger BLOCK_M (amortize weight loading cost across more rows)
```

Things that won't help:
```
✗ Switching to faster tensor core precision (already bandwidth-limited)
✗ Increasing num_warps (SM is already waiting for data)
```

---

## 7. DtoH Sync Is a Silent Killer

### 7.1 The Problem

Every `.item()` or `.cpu()` triggers a GPU→CPU synchronization, stalling until all CUDA ops complete:

```python
# Bad: 32 experts × 5 .item() calls = 160 syncs
for e in range(32):
    count = expert_counts[e].item()     # sync!
    offset = expert_offsets[e].item()   # sync!
    ...
```

### 7.2 The Fix

```python
# Good: batch-fetch all values at once
counts_list = expert_counts.tolist()    # 1 sync
offsets_list = expert_offsets.tolist()   # 1 sync

# Better: no DtoH at all — let the Triton kernel read from GPU memory directly
# (Grouped GEMM: pass expert_offsets as kernel arg, tl.load on the GPU side)
```

---

## 8. Buffer Management

### 8.1 Pre-allocation vs Dynamic Allocation

```python
# Dynamic: simple and safe, but calls cudaMalloc every time
G1_buf = torch.empty((N, 4096), dtype=torch.float32, device=device)

# Pre-allocated: saves malloc, but must manage size and lifetime
# Danger: caching buffers in remote execution environments may cause OOM or crash
# We tried a buffer cache — the remote Modal environment crashed immediately
```

**Recommendation**: Unless the profiler clearly shows malloc as a bottleneck, use dynamic allocation. `torch.empty` is fast under CUDA's caching allocator.

### 8.2 Buffer Elimination Priority

```
O_buf  [N, 7168] FP32  ← largest, eliminate first (fuse scatter into GEMM2)
G1_buf [N, 4096] FP32  ← must keep (GEMM1 output, SwiGLU input)
C_buf  [N, 2048] FP32  ← can eliminate via SwiGLU fusion (but no perf gain in practice)
result [T, 7168] FP32  ← scatter accumulator, eliminable with atomic scatter
```

---

## 9. Sorted Dispatch Pattern

The core of MoE: turn scattered (token, expert) pairs into contiguous per-expert batches.

```
Input: topk_idx [T, TOP_K] — each token selected 8 experts

Step 1: Flatten → T×8 (token_id, expert_id) pairs
Step 2: Filter → keep only local experts (0~31)
Step 3: Sort by expert_id → tokens for the same expert become contiguous
Step 4: unique_consecutive → expert_offsets, expert_counts
Step 5: Gather → A_sorted = A[sorted_token_ids]

Result: each expert maps to a contiguous slice of A_sorted → efficient GEMM
```

**Why sort?** Without sorting, tokens for the same expert are scattered in memory. GEMM degenerates to gather-scatter, tensor cores cannot be used at all.

---

## 10. Debugging Checklist

When numerical results are wrong, check in this order:

```
1. tl.trans()         — Switch to pointer-arithmetic transposition
2. Scale placement    — Applied before or after dot? Block size correct?
3. Precision          — Are intermediates being truncated? (FP8→BF16 vs FP8→FP32)
4. Masks              — Are out-of-bounds elements properly zeroed?
5. Small-scale test   — M=1, K=128, N=128 case vs torch.matmul reference
6. Per-expert test    — Run 1 expert only; confirm single-expert correctness first
7. Strides            — Triton strides are in elements, not bytes
```

---

## 11. Hardware Quick Reference

### B200 (Blackwell, SM100)

| Resource | Value | Kernel Implication |
|----------|-------|--------------------|
| SM count | 192 | Upper bound for persistent kernel grid size |
| FP8 TFLOPS | 9000 | Available for GEMM1 (both operands FP8) |
| TF32 TFLOPS | 2250 | Used by GEMM2 (FP32 A operand) |
| HBM bandwidth | 8 TB/s | The real bottleneck for large GEMMs |
| L2 Cache | 96 MB | Fits ~6-7 experts' GEMM2 weights |
| SMEM/SM | 228 KB | Enough for 4-5 stage pipelining |
| TMEM/SM | 256 KB | Blackwell-only: dedicated accumulator storage |
| Registers/SM | 256 KB | More warps → fewer registers per warp |

### Tensor Core Instruction Evolution

```
Ampere  (SM80):  mma.sync        → synchronous, warp-level (32 threads)
Hopper  (SM90):  wgmma.mma_async → asynchronous, warpgroup-level (128 threads)
Blackwell (SM100): tcgen05.mma   → asynchronous, single-thread dispatch, TMEM accumulation
```

### What tl.dot Compiles To

```python
tl.dot(a_fp8, bt_fp8)
```

On Blackwell compiles to:
```
tcgen05.mma.cta_group::1.kind::f8f6f4  [d-tmem], a-desc, b-desc, idesc
```
- `kind::f8f6f4`: Unified instruction family for FP8/FP6/FP4
- `idesc`: 32-bit descriptor encoding e4m3/e5m2 sub-type and tile dimensions
- `[d-tmem]`: Accumulator lives in TMEM, not general-purpose registers
- Hardware tile: m128 × n128 × k32, single instruction ~11 cycles
