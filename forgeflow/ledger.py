"""Ledger management for tracking optimization trials."""

import csv
from datetime import datetime, timezone
from pathlib import Path

from forgeflow.config import LEDGER_HEADERS, LEDGER_PATH, REPORTS_DIR


def ensure_ledger() -> Path:
    """Create ledger CSV with headers if it doesn't exist."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not LEDGER_PATH.exists():
        with open(LEDGER_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(LEDGER_HEADERS)
    return LEDGER_PATH


def append_ledger_row(
    trial_id: int,
    branch: str,
    commit_sha: str,
    change_summary: str,
    metrics: dict,
    profile: dict | None = None,
    merged: bool = False,
) -> None:
    """Append a single trial row to the ledger CSV."""
    ensure_ledger()
    profile = profile or {}

    row = [
        trial_id,
        datetime.now(timezone.utc).isoformat(),
        branch,
        commit_sha,
        change_summary,
        f"{metrics.get('avg_speedup', 0):.4f}",
        f"{metrics.get('min_speedup', 0):.4f}",
        f"{metrics.get('max_speedup', 0):.4f}",
        f"{metrics.get('avg_latency_ms', 0):.4f}",
        metrics.get("workloads_passed", 0),
        metrics.get("workloads_total", 0),
        profile.get("bottleneck", ""),
        f"{profile.get('sm_throughput_pct', 0):.1f}",
        f"{profile.get('dram_throughput_pct', 0):.1f}",
        f"{profile.get('l1_lsu_pct', 0):.1f}",
        f"{profile.get('tensor_core_pct', 0):.1f}",
        "True" if merged else "False",
    ]

    with open(LEDGER_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def read_ledger() -> list[dict]:
    """Read all ledger entries as list of dicts."""
    ensure_ledger()
    with open(LEDGER_PATH, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_best_trial() -> dict | None:
    """Return the trial with the highest avg_speedup."""
    entries = read_ledger()
    if not entries:
        return None

    def safe_speedup(e: dict) -> float:
        try:
            return float(e.get("avg_speedup", 0))
        except (ValueError, TypeError):
            return 0.0

    return max(entries, key=safe_speedup)


def get_latest_baseline() -> float:
    """Return the avg_speedup from the most recent merged trial, or 0.0."""
    entries = read_ledger()
    merged = [e for e in entries if e.get("merged_to_main") == "True"]
    if not merged:
        return 0.0
    return float(merged[-1].get("avg_speedup", 0))
