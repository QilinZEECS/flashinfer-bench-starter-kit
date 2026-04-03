# NCU Profiling Results

## Files

- `ncu_wl00_v010_fp16_gemm2_detailed.txt` — Full Nsight Compute detailed profile for workload 0 (T=128, smallest workload) on NVIDIA B200.

## How to Run NCU Profiling

### Prerequisites

- Python environment with `modal` installed (`conda activate fi-bench`)
- A Modal account with B200 GPU access
- The `flashinfer-trace` Modal volume populated with the contest trace set

### Run NCU Profile

```bash
# Profile workload 0 with detailed metrics
modal run scripts/run_modal_ncu.py --workload-index 0 --set-name detailed --page details

# Profile a different workload (0-18 available)
modal run scripts/run_modal_ncu.py --workload-index 8 --set-name detailed --page details

# Use "full" set for maximum detail (slower, larger output)
modal run scripts/run_modal_ncu.py --workload-index 0 --set-name full --page details
```

### Available Options

| Flag | Default | Description |
|------|---------|-------------|
| `--workload-index` | 0 | Workload index (0-18). 0=small T, 18=large T |
| `--set-name` | detailed | NCU metric set: `basic`, `detailed`, or `full` |
| `--page` | details | NCU report page: `details`, `raw`, or `summary` |

### How It Works

The script (`scripts/run_modal_ncu.py`) does the following on a remote Modal B200 GPU:

1. Builds a Modal container image with CUDA 12.8.1 dev toolkit, PyTorch 2.8, Triton, and flashinfer-bench
2. Applies two patches to flashinfer-bench:
   - Removes `pin_memory()` calls (unsupported in Modal containers)
   - Removes the hardcoded NVTX filter in `flashinfer_bench/agents/ncu.py` so all kernels are captured (not just NVTX-tagged ones)
3. Packs the solution from `solution/triton/` using `config.toml`
4. Loads the workload from the `flashinfer-trace` Modal volume
5. Runs `flashinfer_bench_run_ncu()` which invokes `ncu` (Nsight Compute CLI) on the kernel
6. Returns the full NCU text output as JSON

### Run Torch Profiler (Alternative)

For a lighter-weight profile showing kernel launch counts and wall-clock times:

```bash
# Profile workload 0
modal run scripts/run_modal_torch_profile.py --workload-index 0

# Profile with more active steps for stable timing
modal run scripts/run_modal_torch_profile.py --workload-index 0 --active-steps 10
```

Reports are saved to `results/profiles/` as both `.json` and `.md` files.

### Interpreting NCU Output

Key metrics to look for:

- **Duration**: Wall-clock time per kernel launch
- **SM Throughput / DRAM Throughput**: How well compute and memory are utilized (higher = better)
- **Achieved Occupancy**: Percentage of maximum warps active (limited by registers, shared memory, or block size)
- **Registers Per Thread**: High values (>128) limit occupancy
- **Shared Memory Per Block**: Combined with registers, determines max blocks per SM
- **L1/L2 Hit Rate**: Cache efficiency
- **Roofline**: Where the kernel sits relative to compute and memory peaks
