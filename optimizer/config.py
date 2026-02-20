"""Configuration constants for the optimizer system."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
KERNEL_PATH = PROJECT_ROOT / "solution" / "triton" / "kernel.py"
KERNEL_BACKUP_PATH = PROJECT_ROOT / "solution" / "triton" / "kernel.py.bak"
SOLUTION_JSON_PATH = PROJECT_ROOT / "solution.json"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
LEDGER_PATH = RESULTS_DIR / "ledger.jsonl"
RUNS_DIR = RESULTS_DIR / "runs"
COMPARISONS_DIR = RESULTS_DIR / "comparisons"
REPORTS_DIR = RESULTS_DIR / "reports"
SWEEPS_DIR = RESULTS_DIR / "sweeps"

# Scripts
PACK_SCRIPT = PROJECT_ROOT / "scripts" / "pack_solution.py"
MODAL_SCRIPT = PROJECT_ROOT / "scripts" / "run_modal.py"

# Benchmark configs
QUICK_BENCH = {"warmup_runs": 2, "iterations": 30, "num_trials": 3}
FULL_BENCH = {"warmup_runs": 3, "iterations": 100, "num_trials": 5}

# Kernel constants (DeepSeek-V3/R1 geometry)
E_GLOBAL = 256
E_LOCAL = 32
H = 7168
I = 2048
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
BLOCK = 128
