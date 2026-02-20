# Run 001: baseline (baseline)

## 基本信息
- **Run ID**: 001
- **时间**: 2026-02-20T09:03:42.937669+00:00
- **策略**: baseline
- **分支**: main
- **Commit**: da03797
- **Kernel SHA256**: b44f3620698189f1

## 优化思路
Initial correctness-first implementation with Python expert loop.

## 结果总览
| 指标 | 值 |
|------|-----|
| Passed / Total | 19 / 19 |
| Avg Speedup | 2.25x |
| Geomean Speedup | 2.07x |
| Min Speedup | 1.16x (5e8dc11c...) |
| Max Speedup | 5.17x (e05c6c03...) |

## Per-Workload 结果
| # | UUID (短) | Speedup | Latency (ms) | Ref Latency (ms) | 状态 |
|---|-----------|---------|-------------|-----------------|------|
| 1 | 5e8dc11c... | 1.16x | 38.985 | None | PASSED |
| 2 | 58a34f27... | 1.2x | 29.793 | None | PASSED |
| 3 | 1a4c6ba1... | 1.36x | 15.465 | None | PASSED |
| 4 | 8f1ff9f1... | 1.61x | 9.764 | None | PASSED |
| 5 | e626d3e6... | 1.68x | 9.064 | None | PASSED |
| 6 | 4822167c... | 1.73x | 8.609 | None | PASSED |
| 7 | 74d7ff04... | 1.75x | 8.442 | None | PASSED |
| 8 | 81955b1e... | 1.8x | 8.036 | None | PASSED |
| 9 | fc378037... | 1.8x | 8.05 | None | PASSED |
| 10 | 76010cb4... | 1.86x | 7.61 | None | PASSED |
| 11 | 6230e838... | 1.99x | 6.981 | None | PASSED |
| 12 | eedc63b2... | 2.08x | 6.502 | None | PASSED |
| 13 | 5eadab1e... | 2.09x | 6.55 | None | PASSED |
| 14 | f7d6ac7c... | 2.23x | 5.917 | None | PASSED |
| 15 | a7c2bcfd... | 2.63x | 4.776 | None | PASSED |
| 16 | 8cba5890... | 2.77x | 4.453 | None | PASSED |
| 17 | b8f4f012... | 3.79x | 3.087 | None | PASSED |
| 18 | 2e69caee... | 3.97x | 2.89 | None | PASSED |
| 19 | e05c6c03... | 5.17x | 2.145 | None | PASSED |

## 分析
Initial correctness-first implementation with Python expert loop.

## 结论与下一步
(待分析)
