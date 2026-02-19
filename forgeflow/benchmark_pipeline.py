"""Benchmark pipeline with auto-rollback on regression."""

import json
from datetime import datetime, timezone
from pathlib import Path

from forgeflow.config import EXPERIMENTS_DIR, MIN_SPEEDUP_IMPROVEMENT_PCT
from forgeflow.git_manager import (
    commit_trial,
    get_current_commit_sha,
    has_uncommitted_changes,
    merge_trial_to_main,
    rollback_trial,
    should_merge,
)
from forgeflow.validate import run_full_validation


def save_experiment(trial_id: int, results: dict, metrics: dict, profile: dict | None = None) -> Path:
    """Save experiment results to experiments/ directory."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    experiment_path = EXPERIMENTS_DIR / f"trial_{trial_id}_{timestamp}.json"

    experiment_data = {
        "trial_id": trial_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit_sha": get_current_commit_sha(),
        "metrics": metrics,
        "profile": profile or {},
        "workload_results": results,
    }

    experiment_path.write_text(json.dumps(experiment_data, indent=2, default=str))
    return experiment_path


def run_pipeline(
    trial_id: int,
    change_summary: str,
    baseline_avg_speedup: float,
) -> tuple[bool, dict]:
    """Execute the full benchmark pipeline for a trial.

    Steps:
        1. Validate correctness + collect metrics
        2. Compare against baseline
        3. Commit if improved, rollback if regressed

    Returns (success, metrics).
    """
    print(f"\n{'='*60}")
    print(f"  ForgeFlow Pipeline — Trial {trial_id}")
    print(f"  Change: {change_summary}")
    print(f"{'='*60}\n")

    # Step 1: Full validation
    try:
        all_passed, results, metrics = run_full_validation()
    except Exception as e:
        print(f"[pipeline] ABORT: Benchmark failed with error: {e}")
        rollback_trial()
        return False, {}

    # Step 2: Save experiment data
    save_experiment(trial_id, results, metrics)

    # Step 3: Check correctness
    if not all_passed:
        print(f"[pipeline] ROLLBACK: Correctness check failed "
              f"({metrics['workloads_passed']}/{metrics['workloads_total']})")
        rollback_trial()
        return False, metrics

    # Step 4: Check performance regression
    new_avg = metrics["avg_speedup"]
    if baseline_avg_speedup > 0:
        change_pct = ((new_avg - baseline_avg_speedup) / baseline_avg_speedup) * 100
        print(f"[pipeline] Performance delta: {change_pct:+.2f}% "
              f"(baseline={baseline_avg_speedup:.2f}x → new={new_avg:.2f}x)")

        if change_pct < -1.0:
            print(f"[pipeline] ROLLBACK: Performance regression detected ({change_pct:.1f}%)")
            rollback_trial()
            return False, metrics
    else:
        print(f"[pipeline] First run, no baseline to compare")

    # Step 5: Commit on trial branch
    if has_uncommitted_changes():
        sha = commit_trial(trial_id, change_summary, metrics)
        print(f"[pipeline] Committed on trial branch: {sha}")

    # Step 6: Merge to main if improvement is significant
    merged = False
    if should_merge(baseline_avg_speedup, new_avg):
        print(f"[pipeline] MERGE: Improvement > {MIN_SPEEDUP_IMPROVEMENT_PCT}%, merging to main")
        merge_trial_to_main(trial_id)
        merged = True
    else:
        print(f"[pipeline] SKIP MERGE: Improvement below {MIN_SPEEDUP_IMPROVEMENT_PCT}% threshold")

    return True, {**metrics, "merged_to_main": merged}
