# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Entry for the **FlashInfer AI Kernel Generation Contest @ MLSys 2026**, `fused_moe` track (FP8 block-scale, DeepSeek-V3/R1). Write a Triton kernel on NVIDIA B200 targeting 8x+ speedup over the reference implementation.

## Commands

```bash
# Activate environment
conda activate fi-bench

# Pack kernel into solution.json
python scripts/pack_solution.py

# Benchmark on Modal B200 (the only way to test — no local GPU)
modal run scripts/run_modal.py
```

The full dev loop is: edit `solution/triton/kernel.py` → pack → benchmark on Modal.

## Architecture

The only file to modify is `solution/triton/kernel.py`. Everything else is infrastructure.

### Kernel Entry Point

The `kernel` function uses **Destination Passing Style** (DPS): 10 inputs + 1 pre-allocated output as the last parameter. The function writes results into `output` (no return value).

### Parameter Layout

| Parameter | Shape | Dtype |
|-----------|-------|-------|
| routing_logits | [T, 256] | float32 |
| routing_bias | [256] | bfloat16 |
| hidden_states | [T, 7168] | float8_e4m3fn |
| hidden_states_scale | **[56, T]** | float32 |
| gemm1_weights | [32, 4096, 7168] | float8_e4m3fn |
| gemm1_weights_scale | [32, 32, 56] | float32 |
| gemm2_weights | [32, 7168, 2048] | float8_e4m3fn |
| gemm2_weights_scale | [32, 56, 16] | float32 |
| local_expert_offset | scalar | int |
| routed_scaling_factor | scalar | float |
| **output** (DPS) | [T, 7168] | bfloat16 |

Constants: E_global=256, E_local=32, H=7168, I=2048, TOP_K=8, N_GROUP=8, TOPK_GROUP=4, BLOCK=128.

### Algorithm (current correctness-first impl)

1. **Routing**: sigmoid scores (not softmax), bias added for selection only, top-4 groups of 8, top-8 experts within groups, weights normalized with unbiased sigmoid scores × `routed_scaling_factor`
2. **Per-expert loop** (Python for over 32 experts): dequant FP8 → GEMM1 → SwiGLU (`silu(gate)*up`, first I cols=up, last I cols=gate) → GEMM2 → weighted accumulate
3. **Output**: `output.copy_(result.to(bfloat16))`

### Critical Details

- `hidden_states_scale` is **transposed**: `[num_h_blocks, T]` not `[T, num_h_blocks]`
- `local_expert_offset` and `routed_scaling_factor` may arrive as 0-dim tensors — use `.item()`
- SwiGLU split: first I columns = up (X1), last I columns = gate (X2)

### Baseline Performance

19 workloads all PASSED. avg ~2.2x, min 1.16x, max 5.04x. Small T fast (3-5x), large T slow (1.1-1.6x). Primary bottleneck: Python expert loop causing 128+ kernel launches.
