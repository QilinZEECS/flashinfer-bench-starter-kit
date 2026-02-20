# Run 003: v002_fp8_scaled_mm (P1)

## 基本信息
- **Run ID**: 003
- **时间**: 2026-02-20T09:48:17.859393+00:00
- **策略**: P1
- **分支**: opt/v002-fp8-scaled-mm
- **Commit**: 134430c
- **Kernel SHA256**: b3ee32b545d63cb0

## 优化思路
V002: sorted dispatch + torch._scaled_mm for FP8 tensor cores

## 结果总览
| 指标 | 值 |
|------|-----|
| Passed / Total | 19 / 19 |
| Avg Speedup | 0.47x |
| Geomean Speedup | 0.45x |
| Min Speedup | 0.33x (8f1ff9f1...) |
| Max Speedup | 1.03x (e05c6c03...) |

## Per-Workload 结果
| # | UUID (短) | Speedup | Latency (ms) | Ref Latency (ms) | 状态 |
|---|-----------|---------|-------------|-----------------|------|
| 1 | 8f1ff9f1... | 0.33x | 46.807 | None | PASSED |
| 2 | e626d3e6... | 0.34x | 44.349 | None | PASSED |
| 3 | 4822167c... | 0.34x | 43.817 | None | PASSED |
| 4 | 74d7ff04... | 0.35x | 42.282 | None | PASSED |
| 5 | 81955b1e... | 0.35x | 40.515 | None | PASSED |
| 6 | 76010cb4... | 0.35x | 39.808 | None | PASSED |
| 7 | fc378037... | 0.35x | 41.201 | None | PASSED |
| 8 | 6230e838... | 0.36x | 38.332 | None | PASSED |
| 9 | 1a4c6ba1... | 0.36x | 58.329 | None | PASSED |
| 10 | eedc63b2... | 0.39x | 34.176 | None | PASSED |
| 11 | 5eadab1e... | 0.41x | 33.033 | None | PASSED |
| 12 | f7d6ac7c... | 0.42x | 30.942 | None | PASSED |
| 13 | 58a34f27... | 0.49x | 73.255 | None | PASSED |
| 14 | a7c2bcfd... | 0.51x | 24.964 | None | PASSED |
| 15 | 8cba5890... | 0.52x | 23.289 | None | PASSED |
| 16 | 5e8dc11c... | 0.54x | 82.589 | None | PASSED |
| 17 | b8f4f012... | 0.72x | 16.076 | None | PASSED |
| 18 | 2e69caee... | 0.84x | 13.491 | None | PASSED |
| 19 | e05c6c03... | 1.03x | 10.637 | None | PASSED |

## 与基线的对比 (vs Run 002)
| 指标 | Baseline | This Run | Delta |
|------|----------|----------|-------|
| Geomean | 2.5x | 0.45x | -2.05x |

## 分析
V002: sorted dispatch + torch._scaled_mm for FP8 tensor cores

## 结论与下一步
(待分析)
