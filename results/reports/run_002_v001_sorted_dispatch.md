# Run 002: v001_sorted_dispatch (P0)

## 基本信息
- **Run ID**: 002
- **时间**: 2026-02-20T09:10:30.845196+00:00
- **策略**: P0
- **分支**: opt/v001-sorted-dispatch
- **Commit**: 7bdf3e2
- **Kernel SHA256**: ae37ce07c4e5ca41

## 优化思路
Sorted dispatch: flatten token-expert pairs, sort by expert, batch process with contiguous access. Eliminates Python for-loop overhead.

## 结果总览
| 指标 | 值 |
|------|-----|
| Passed / Total | 19 / 19 |
| Avg Speedup | 2.89x |
| Geomean Speedup | 2.5x |
| Min Speedup | 1.2x (5e8dc11c...) |
| Max Speedup | 8.68x (e05c6c03...) |

## Per-Workload 结果
| # | UUID (短) | Speedup | Latency (ms) | Ref Latency (ms) | 状态 |
|---|-----------|---------|-------------|-----------------|------|
| 1 | 5e8dc11c... | 1.2x | 37.642 | None | PASSED |
| 2 | 58a34f27... | 1.27x | 28.289 | None | PASSED |
| 3 | 1a4c6ba1... | 1.49x | 14.137 | None | PASSED |
| 4 | 8f1ff9f1... | 1.82x | 8.691 | None | PASSED |
| 5 | e626d3e6... | 1.92x | 7.893 | None | PASSED |
| 6 | 4822167c... | 1.99x | 7.467 | None | PASSED |
| 7 | 74d7ff04... | 2.01x | 7.317 | None | PASSED |
| 8 | 81955b1e... | 2.09x | 6.894 | None | PASSED |
| 9 | fc378037... | 2.09x | 6.926 | None | PASSED |
| 10 | 76010cb4... | 2.17x | 6.519 | None | PASSED |
| 11 | 6230e838... | 2.3x | 6.031 | None | PASSED |
| 12 | eedc63b2... | 2.47x | 5.462 | None | PASSED |
| 13 | 5eadab1e... | 2.5x | 5.471 | None | PASSED |
| 14 | f7d6ac7c... | 2.73x | 4.826 | None | PASSED |
| 15 | a7c2bcfd... | 3.29x | 3.83 | None | PASSED |
| 16 | 8cba5890... | 3.54x | 3.474 | None | PASSED |
| 17 | b8f4f012... | 5.32x | 2.191 | None | PASSED |
| 18 | 2e69caee... | 6.0x | 1.912 | None | PASSED |
| 19 | e05c6c03... | 8.68x | 1.282 | None | PASSED |

## 分析
Sorted dispatch: flatten token-expert pairs, sort by expert, batch process with contiguous access. Eliminates Python for-loop overhead.

## 结论与下一步
(待分析)
