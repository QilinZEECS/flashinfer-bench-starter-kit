# Run 004: v003_fp16_gemm (P1)

## 基本信息
- **Run ID**: 004
- **时间**: 2026-02-20T09:54:37.058799+00:00
- **策略**: P1
- **分支**: opt/v003-fp16-gemm
- **Commit**: 2da1e5b
- **Kernel SHA256**: a9b43c5d0cc800c3

## 优化思路
V003: sorted dispatch + FP16 dequant/GEMM. Uses FP16 tensor cores for both GEMMs, halves memory bandwidth. SwiGLU in float32 internally.

## 结果总览
| 指标 | 值 |
|------|-----|
| Passed / Total | 0 / 19 |
| Avg Speedup | 0.0x |
| Geomean Speedup | 0.0x |

## Per-Workload 结果
| # | UUID (短) | Speedup | Latency (ms) | Ref Latency (ms) | 状态 |
|---|-----------|---------|-------------|-----------------|------|
| 1 | b8f4f012... | Nonex | None | None | INCORRECT_NUMERICAL |
| 2 | e05c6c03... | Nonex | None | None | INCORRECT_NUMERICAL |
| 3 | 6230e838... | Nonex | None | None | INCORRECT_NUMERICAL |
| 4 | 8f1ff9f1... | Nonex | None | None | INCORRECT_NUMERICAL |
| 5 | 1a4c6ba1... | Nonex | None | None | INCORRECT_NUMERICAL |
| 6 | a7c2bcfd... | Nonex | None | None | INCORRECT_NUMERICAL |
| 7 | 2e69caee... | Nonex | None | None | INCORRECT_NUMERICAL |
| 8 | 8cba5890... | Nonex | None | None | INCORRECT_NUMERICAL |
| 9 | 5e8dc11c... | Nonex | None | None | INCORRECT_NUMERICAL |
| 10 | 58a34f27... | Nonex | None | None | INCORRECT_NUMERICAL |
| 11 | 5eadab1e... | Nonex | None | None | INCORRECT_NUMERICAL |
| 12 | eedc63b2... | Nonex | None | None | INCORRECT_NUMERICAL |
| 13 | e626d3e6... | Nonex | None | None | INCORRECT_NUMERICAL |
| 14 | 74d7ff04... | Nonex | None | None | INCORRECT_NUMERICAL |
| 15 | 4822167c... | Nonex | None | None | INCORRECT_NUMERICAL |
| 16 | 81955b1e... | Nonex | None | None | INCORRECT_NUMERICAL |
| 17 | 76010cb4... | Nonex | None | None | INCORRECT_NUMERICAL |
| 18 | fc378037... | Nonex | None | None | INCORRECT_NUMERICAL |
| 19 | f7d6ac7c... | Nonex | None | None | INCORRECT_NUMERICAL |

## 与基线的对比 (vs Run 002)
| 指标 | Baseline | This Run | Delta |
|------|----------|----------|-------|
| Geomean | 2.5x | 0.0x | -2.50x |

## 分析
V003: sorted dispatch + FP16 dequant/GEMM. Uses FP16 tensor cores for both GEMMs, halves memory bandwidth. SwiGLU in float32 internally.

## 结论与下一步
(待分析)
