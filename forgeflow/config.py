"""ForgeFlow configuration constants."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
SOLUTION_DIR = PROJECT_ROOT / "solution" / "triton"
KERNEL_PATH = SOLUTION_DIR / "kernel.py"
SOLUTION_JSON = PROJECT_ROOT / "solution.json"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"
LEDGER_PATH = REPORTS_DIR / "ledger.csv"

# ── Git ──────────────────────────────────────────────────────────────
MAIN_BRANCH = "main"
TRIAL_BRANCH_PREFIX = "opt/trial-"
MIN_SPEEDUP_IMPROVEMENT_PCT = 2.0  # merge to main only if > 2% gain

# ── Benchmark ────────────────────────────────────────────────────────
MODAL_GPU = "B200:1"
BENCHMARK_WARMUP = 3
BENCHMARK_ITERATIONS = 100
BENCHMARK_TRIALS = 5
DEFINITION_NAME = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"

# ── Performance Targets ──────────────────────────────────────────────
SPEEDUP_TARGETS = {
    "minimum": 3.0,       # all workloads should reach
    "competitive": 6.0,   # strong submission
    "goal": 8.0,          # target for T=4096
}

# ── NCU Metrics of Interest ──────────────────────────────────────────
NCU_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sectors_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
]

# ── Bottleneck Classification Thresholds ─────────────────────────────
COMPUTE_BOUND_THRESHOLD = 60   # sm throughput > 60% → compute-bound
MEMORY_BOUND_THRESHOLD = 60    # dram throughput > 60% → memory-bound
TENSOR_CORE_UTIL_LOW = 30      # tensor pipe < 30% → tensor cores underused

# ── Ledger CSV Headers ───────────────────────────────────────────────
LEDGER_HEADERS = [
    "trial_id",
    "timestamp",
    "branch",
    "commit_sha",
    "change_summary",
    "avg_speedup",
    "min_speedup",
    "max_speedup",
    "avg_latency_ms",
    "workloads_passed",
    "workloads_total",
    "bottleneck",
    "sm_throughput_pct",
    "dram_throughput_pct",
    "l1_lsu_pct",
    "tensor_core_pct",
    "merged_to_main",
]
