# Run 002 Detailed Profiling Report (2026-03-02)

## Scope
- Variant: `v001_sorted_dispatch` (commit `7bdf3e2`, kernel hash `ae37ce07c4e5ca41`)
- Goal: provide a deeper profiling report to guide the next optimization round.

## Data Sources
- Benchmark summary/results: `results/runs/run_002.json`
- Workload metadata: `mlsys26-contest/workloads/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.jsonl`
- Fresh Modal profiling runs (2026-03-02):
  - NCU on real workload: `ap-mvQm47tip7m4BDSgx35Hfi`
  - NCU torch sanity: `ap-8YfbaC3lcBtJhTqAx0N5pB`
  - NCU pure CUDA sanity: `ap-xT3Uonl3gvEi5Tex1iL1kg`

## Executive Summary
- Run 002 performance is solid overall (`avg 2.89x`, `geomean 2.5x`, `19/19 passed`), but long-sequence cases remain the primary bottleneck.
- NCU cannot currently capture kernels on Modal in this setup:
  - `LibraryNotLoaded` on CUDA 12.8 torch sanity.
  - Return code `11` with `No kernels were profiled` on both real workload and pure CUDA sanity.
- Because hardware counters are unavailable, this report uses benchmark-level profiling + code-path analysis to prioritize improvements.

## Modal NCU Status (Current Blocker)
1. Real workload profiling (`run_modal_ncu.py`, workload index 0) failed with:
   - `NCU exited with non-zero return code 11`
   - `No kernels were profiled`
   - segfault stack rooted in CUDA stream/context init (`cuDevicePrimaryCtxRetain`, `c10::cuda::*`)
2. Torch sanity profiling failed with:
   - `Failed to initialize the profiler: LibraryNotLoaded`
3. Pure CUDA binary sanity:
   - binary runs normally (`plain_returncode=0`)
   - both `--target-processes application-only` and `--target-processes all` fail with return code `11` and no kernels.

Inference: this is likely an environment/runtime profiler issue on Modal for the current image/driver/profiler combination, not a bug in your kernel command line.

### NCU command matrix check (2026-03-03)
Command matrix was run on pure CUDA `saxpy` to test whether `detailed` is unavailable.

| Mode | Command | Result |
|---|---|---|
| list sets | `ncu --list-sets` | success, shows `basic`, `detailed`, `full`, `roofline`, ... |
| default | `ncu --target-processes all ./saxpy` | return code 11, no kernels profiled |
| detailed | `ncu --set detailed --target-processes all ./saxpy` | return code 11, no kernels profiled |
| full | `ncu --set full --target-processes all ./saxpy` | return code 11, no kernels profiled |
| section only | `ncu --section LaunchStats --target-processes all ./saxpy` | return code 11, no kernels profiled |
| speedOfLight | `ncu --set speedOfLight --target-processes all ./saxpy` | return code 11, no kernels profiled |
| app-only | `ncu --set detailed --target-processes application-only ./saxpy` | return code 11, no kernels profiled |
| cache control | `ncu --set detailed --cache-control none --target-processes all ./saxpy` | return code 11, no kernels profiled |

Conclusion: `detailed` is available; crash is not caused by set/page choice.

## Workload-Level Profile (Run 002)

### Top latency hotspots
| UUID (short) | seq_len | latency (ms) | speedup | max_rel_error |
|---|---:|---:|---:|---:|
| `5e8dc11c` | 14107 | 37.642 | 1.20 | 0.50 |
| `58a34f27` | 11948 | 28.289 | 1.27 | 0.69 |
| `1a4c6ba1` | 901 | 14.137 | 1.49 | 0.333 |
| `8f1ff9f1` | 80 | 8.691 | 1.82 | 0.00758 |
| `e626d3e6` | 58 | 7.893 | 1.92 | 0.00741 |

### Seq length scaling view
| Seq_len bin | # workloads | avg latency (ms) | avg speedup | geomean speedup |
|---|---:|---:|---:|---:|
| 1-16 | 5 | 2.538 | 5.366 | 5.032 |
| 17-128 | 11 | 6.682 | 2.190 | 2.174 |
| 129-2048 | 1 | 14.137 | 1.490 | 1.490 |
| 2049-20000 | 2 | 32.966 | 1.235 | 1.235 |

- Pearson correlation (`seq_len` vs `latency`) = `0.9555` (very strong).
- Main conclusion: large-`seq_len` throughput is the dominant optimization target.

## Code-Path Bottleneck Mapping
Reference file: `solution/triton/kernel.py` (sorted-dispatch branch at `7bdf3e2`).

1. Expert weight dequantization is done inside per-expert compute loop.
   - See dequant calls in loop: lines around `255-262`.
   - Cost scales with number of active experts per invocation.
2. Two full GEMMs are issued per expert chunk (`GEMM1`, `GEMM2`).
   - Lines around `264-271`.
   - For large sequence lengths, this dominates total compute and memory traffic.
3. Accumulation uses `index_add_` scatter-style updates.
   - Line around `274`.
   - Potential write amplification/atomics contention pattern depending on routing distribution.
4. Dispatch preprocessing (`argsort`, `unique_consecutive`) adds non-trivial overhead.
   - Lines around `224-233`.
   - Especially visible when token-expert pair count is very large.

## Prioritized Optimization Roadmap
1. Cache dequantized expert weights within an invocation (or persistent cache across iterations with invalidation on weight pointer/version change).
   - Why: removes repeated FP8->FP32 expansion work in the hot loop.
   - Expected impact: medium-high on long sequences.
2. Move from per-expert looped GEMM toward grouped/batched GEMM path.
   - Why: better GPU occupancy and launch amortization.
   - Expected impact: high for mid/long sequence bins.
3. Replace dense `weights [T, E_GLOBAL]` materialization with sparse top-k representation.
   - Why: reduce memory footprint and bandwidth in routing/weight gather.
   - Expected impact: medium; helps both latency and memory pressure.
4. Reduce sort/dispatch overhead.
   - Try radix/bucket dispatch by local expert instead of full sort when possible.
   - Expected impact: medium for very large token counts.
5. Numeric stability pass for long-sequence error tail (`max_rel_error` up to `0.69`).
   - Focus workloads: `58a34f27`, `5e8dc11c`, `1a4c6ba1`.
   - Check scaling/accumulation precision path before any aggressive fusion changes.

## NCU Enablement Plan (So We Can Collect True Kernel Counters)
1. Keep current benchmark flow, but run profiler on a minimal CUDA image where NCU+driver is known compatible (single process, single kernel).
2. Once minimal NCU capture succeeds, reuse same image pin for benchmark app.
3. Capture at least one `.ncu-rep` on a short-seq workload first; then collect long-seq hotspot.
4. Export `--page details` and `--page raw --csv` to compare:
   - SM busy / occupancy
   - DRAM throughput
   - L2/L1 traffic
   - stall reasons (memory dependency, not selected, barrier, etc.)

## Recommended Next Experiment
1. Implement weight dequant cache + sparse top-k weights path.
2. Re-run full benchmark and compare long-seq workloads (`5e8dc11c`, `58a34f27`, `1a4c6ba1`) first.
3. If NCU remains blocked on Modal, run Nsight Compute on a non-Modal environment with identical kernel code to get counter-guided tuning.

## 2026-03-04 Optimization Update
- Follow-up variant: `v004_sorted_dispatch_plus_cache`.
- Report: `results/reports/run_005_v004_sorted_dispatch_plus_cache.md`.
- Kernel changes applied in `solution/triton/kernel.py`:
  - Added process-local dequant cache (`_WEIGHT_CACHE_KEY`, `_WEIGHT_CACHE_W13`, `_WEIGHT_CACHE_W2`) and `_get_weight_cache(...)`.
  - Reused dequantized expert weights across repeated kernel invocations.
  - Reduced host-device scalar sync overhead by materializing `unique_experts` and `expert_counts` once via `tolist()`.
- Full benchmark outcome after changes:
  - `19/19 PASSED`
  - `avg speedup 3.30x`, `geomean 2.90x`
  - Prior local full-run reference before this update: `avg ~2.19x`, `geomean ~2.05x`
- Remaining bottleneck remains long-sequence workloads (`5e8dc11c`, `58a34f27`), still near `1.27x-1.36x`.
