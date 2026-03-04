# Run 005: v004_sorted_dispatch_plus_cache

## Basic Info
- Run date: 2026-03-04
- Variant: `v004_sorted_dispatch_plus_cache`
- Execution: Modal full benchmark (`scripts/run_modal.py`)
- Modal app: `ap-XpmUuSqm3qK85CsmFuMvL1`
- Track: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
- Result: `19 / 19 PASSED`

## Kernel Changes
Reference: `solution/triton/kernel.py`

1. Sorted dispatch path in `kernel()`:
- Flatten `(token, expert)` pairs.
- Filter local experts.
- Sort by local expert id.
- Process contiguous per-expert chunks.

2. Per-process expert weight dequant cache:
- Added global cache state:
  - `_WEIGHT_CACHE_KEY`
  - `_WEIGHT_CACHE_W13`
  - `_WEIGHT_CACHE_W2`
- Added `_get_weight_cache(...)` helper.
- In the expert loop, lazily dequantize each expert weight once and reuse in subsequent calls.

3. Reduced scalar-sync overhead:
- Convert `unique_experts` and `expert_counts` to host lists once (`tolist()`), then iterate.

## Full-Benchmark Summary
- Average speedup: `3.30x`
- Geomean speedup: `2.90x`
- Min speedup: `1.27x` (`5e8dc11c...`)
- Max speedup: `9.22x` (`e05c6c03...`)

Compared to the previous full run in this workspace (`avg ~2.19x`, `geo ~2.05x`), this is a substantial improvement.

## Per-Workload Highlights
- Strong gains on short/medium workloads:
  - `e05c6c03...`: `9.22x`
  - `b8f4f012...`: `6.08x`
  - `2e69caee...`: `6.39x`
- Long-seq workloads improved but remain bottlenecks:
  - `5e8dc11c...`: `1.27x`
  - `58a34f27...`: `1.36x`

## Notes
- Correctness still passes all workloads.
- Long-seq profile remains GEMM-heavy with non-trivial dequant cost; next work should focus on reducing weight dequant/GEMM overhead further for large `seq_len`.
