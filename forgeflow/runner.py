"""ForgeFlow Runner: Main orchestrator for the self-iterating optimization loop.

Usage:
    python -m forgeflow.runner baseline     # Run initial baseline measurement
    python -m forgeflow.runner trial        # Run one optimization trial
    python -m forgeflow.runner loop [N]     # Run N trials (default: 5)
    python -m forgeflow.runner status       # Print current ledger summary
"""

import argparse

from forgeflow.benchmark_pipeline import run_pipeline, save_experiment
from forgeflow.config import SPEEDUP_TARGETS
from forgeflow.git_manager import (
    create_trial_branch,
    ensure_repo_initialized,
    get_current_branch,
    get_current_commit_sha,
    get_next_trial_id,
    rollback_trial,
)
from forgeflow.hardware_oracle import (
    ProfileResult,
    print_profile_report,
    run_ncu_profile,
)
from forgeflow.ledger import (
    append_ledger_row,
    ensure_ledger,
    get_best_trial,
    get_latest_baseline,
    read_ledger,
)
from forgeflow.validate import run_full_validation


def run_baseline() -> dict:
    """Run baseline measurement without any code changes.

    Returns metrics dict from the baseline run.
    """
    print("\n" + "=" * 60)
    print("  ForgeFlow — Baseline Measurement")
    print("=" * 60 + "\n")

    ensure_repo_initialized()
    ensure_ledger()

    print("[runner] Running full validation on current kernel...")
    try:
        all_passed, results, metrics = run_full_validation()
    except Exception as e:
        print(f"[runner] FATAL: Baseline benchmark failed: {e}")
        return {}

    if not all_passed:
        print(f"[runner] WARNING: Baseline has correctness failures "
              f"({metrics['workloads_passed']}/{metrics['workloads_total']})")

    # Save as trial 0 (baseline)
    save_experiment(0, results, metrics)
    branch = get_current_branch()
    sha = get_current_commit_sha()

    append_ledger_row(
        trial_id=0,
        branch=branch,
        commit_sha=sha,
        change_summary="baseline measurement",
        metrics=metrics,
        merged=True,
    )

    _print_metrics_summary(metrics, "Baseline")
    return metrics


def run_trial(change_summary: str = "optimization trial") -> tuple[bool, dict]:
    """Run a single optimization trial with full pipeline.

    Steps:
        1. Get next trial ID
        2. Create trial branch
        3. Run benchmark pipeline (validate → compare → commit/rollback)
        4. Record to ledger

    Returns (success, metrics).
    """
    ensure_repo_initialized()
    ensure_ledger()

    trial_id = get_next_trial_id()
    baseline_avg = get_latest_baseline()

    print(f"\n[runner] Starting trial {trial_id} "
          f"(baseline avg_speedup={baseline_avg:.2f}x)")

    # Create trial branch
    branch_name = create_trial_branch(trial_id)
    print(f"[runner] Created branch: {branch_name}")

    # Run full pipeline
    try:
        success, metrics = run_pipeline(trial_id, change_summary, baseline_avg)
    except Exception as e:
        print(f"[runner] Trial {trial_id} failed unexpectedly: {e}")
        rollback_trial()
        return False, {}

    # Record to ledger
    sha = get_current_commit_sha()
    branch = get_current_branch()
    merged = metrics.get("merged_to_main", False)

    append_ledger_row(
        trial_id=trial_id,
        branch=branch,
        commit_sha=sha,
        change_summary=change_summary,
        metrics=metrics,
        merged=merged,
    )

    if success:
        _print_metrics_summary(metrics, f"Trial {trial_id}")
    else:
        print(f"[runner] Trial {trial_id} FAILED — rolled back")

    return success, metrics


def run_profiled_trial(change_summary: str = "profiled optimization trial") -> tuple[bool, dict, ProfileResult]:
    """Run a trial with NCU profiling for hardware-aware feedback.

    Returns (success, metrics, profile).
    """
    ensure_repo_initialized()
    ensure_ledger()

    trial_id = get_next_trial_id()
    baseline_avg = get_latest_baseline()

    print(f"\n[runner] Starting profiled trial {trial_id}")

    # Step 1: Profile current kernel
    print("[runner] Running NCU profiling...")
    profile = run_ncu_profile()
    print_profile_report(profile)

    # Step 2: Create trial branch and run pipeline
    branch_name = create_trial_branch(trial_id)
    print(f"[runner] Created branch: {branch_name}")

    try:
        success, metrics = run_pipeline(trial_id, change_summary, baseline_avg)
    except Exception as e:
        print(f"[runner] Profiled trial {trial_id} failed unexpectedly: {e}")
        rollback_trial()
        return False, {}, profile

    # Step 3: Record to ledger with profile data
    sha = get_current_commit_sha()
    branch = get_current_branch()
    merged = metrics.get("merged_to_main", False)

    append_ledger_row(
        trial_id=trial_id,
        branch=branch,
        commit_sha=sha,
        change_summary=change_summary,
        metrics=metrics,
        profile=profile.to_dict(),
        merged=merged,
    )

    if success:
        _print_metrics_summary(metrics, f"Trial {trial_id} (profiled)")

    return success, metrics, profile


def run_loop(max_trials: int = 5) -> None:
    """Run the self-iteration loop for up to max_trials.

    Loop stops early if the goal speedup target is reached.
    """
    print("\n" + "=" * 60)
    print(f"  ForgeFlow — Self-Iteration Loop ({max_trials} trials)")
    print("=" * 60 + "\n")

    goal = SPEEDUP_TARGETS["goal"]
    best_avg = get_latest_baseline()

    if best_avg == 0:
        print("[runner] No baseline found. Running baseline first...")
        baseline_metrics = run_baseline()
        best_avg = baseline_metrics.get("avg_speedup", 0)

    for i in range(1, max_trials + 1):
        print(f"\n{'─' * 60}")
        print(f"  Iteration {i}/{max_trials}  |  Best avg_speedup: {best_avg:.2f}x  |  Goal: {goal:.1f}x")
        print(f"{'─' * 60}")

        if best_avg >= goal:
            print(f"\n[runner] GOAL REACHED: {best_avg:.2f}x >= {goal:.1f}x target!")
            break

        success, metrics = run_trial(f"auto-iteration {i}")

        if success:
            new_avg = metrics.get("avg_speedup", 0)
            if new_avg > best_avg:
                best_avg = new_avg
                print(f"[runner] New best: {best_avg:.2f}x")
        else:
            print(f"[runner] Trial failed, continuing...")

    _print_final_summary()


def print_status() -> None:
    """Print current optimization status from the ledger."""
    entries = read_ledger()

    print(f"\n{'=' * 70}")
    print(f"  ForgeFlow — Optimization Status")
    print(f"{'=' * 70}")

    if not entries:
        print("  No trials recorded yet. Run 'baseline' first.")
        print(f"{'=' * 70}\n")
        return

    print(f"  Total trials: {len(entries)}")

    best = get_best_trial()
    if best:
        print(f"  Best trial:   #{best['trial_id']} — "
              f"{float(best['avg_speedup']):.2f}x avg speedup")

    merged_count = sum(1 for e in entries if e.get("merged_to_main") == "True")
    print(f"  Merged to main: {merged_count}/{len(entries)}")

    print(f"\n  {'Trial':>6} {'Branch':<20} {'Avg':>8} {'Min':>8} {'Max':>8} "
          f"{'Pass':>6} {'Merged':>7}")
    print(f"  {'─' * 6} {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 6} {'─' * 7}")

    for entry in entries:
        tid = entry.get("trial_id", "?")
        branch = entry.get("branch", "?")[:20]
        avg_s = float(entry.get("avg_speedup", 0))
        min_s = float(entry.get("min_speedup", 0))
        max_s = float(entry.get("max_speedup", 0))
        passed = entry.get("workloads_passed", "?")
        total = entry.get("workloads_total", "?")
        merged = "yes" if entry.get("merged_to_main") == "True" else "no"
        print(f"  {tid:>6} {branch:<20} {avg_s:>7.2f}x {min_s:>7.2f}x "
              f"{max_s:>7.2f}x {passed:>3}/{total:<2} {merged:>7}")

    # Show targets
    current_best = float(best["avg_speedup"]) if best else 0
    print(f"\n  Targets:")
    for name, target in SPEEDUP_TARGETS.items():
        status = "REACHED" if current_best >= target else f"{current_best:.2f}/{target:.1f}x"
        print(f"    {name:<12}: {status}")

    print(f"{'=' * 70}\n")


def _print_metrics_summary(metrics: dict, label: str) -> None:
    """Print a compact metrics summary."""
    print(f"\n  {label} Results:")
    print(f"    avg_speedup: {metrics.get('avg_speedup', 0):.2f}x")
    print(f"    min_speedup: {metrics.get('min_speedup', 0):.2f}x")
    print(f"    max_speedup: {metrics.get('max_speedup', 0):.2f}x")
    print(f"    avg_latency: {metrics.get('avg_latency_ms', 0):.2f}ms")
    print(f"    correctness: {metrics.get('workloads_passed', 0)}"
          f"/{metrics.get('workloads_total', 0)} passed")


def _print_final_summary() -> None:
    """Print final summary after a loop completes."""
    best = get_best_trial()
    if not best:
        print("\n[runner] No successful trials recorded.")
        return

    print(f"\n{'=' * 60}")
    print(f"  ForgeFlow — Final Summary")
    print(f"{'=' * 60}")
    print(f"  Best trial: #{best['trial_id']}")
    print(f"  Avg speedup: {float(best['avg_speedup']):.2f}x")
    print(f"  Min speedup: {float(best['min_speedup']):.2f}x")
    print(f"  Max speedup: {float(best['max_speedup']):.2f}x")

    goal = SPEEDUP_TARGETS["goal"]
    current = float(best["avg_speedup"])
    if current >= goal:
        print(f"  Status: GOAL REACHED ({current:.2f}x >= {goal:.1f}x)")
    else:
        gap = ((goal - current) / goal) * 100
        print(f"  Status: {gap:.1f}% below goal ({current:.2f}x / {goal:.1f}x)")
    print(f"{'=' * 60}\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ForgeFlow: Self-iterating kernel optimization runner",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("baseline", help="Run baseline measurement")

    trial_parser = subparsers.add_parser("trial", help="Run one optimization trial")
    trial_parser.add_argument(
        "-m", "--message", default="manual trial",
        help="Change summary for this trial",
    )

    loop_parser = subparsers.add_parser("loop", help="Run N optimization trials")
    loop_parser.add_argument(
        "n", type=int, nargs="?", default=5,
        help="Number of trials to run (default: 5)",
    )

    subparsers.add_parser("status", help="Print optimization status")
    subparsers.add_parser("profile", help="Run a profiled trial with NCU")

    args = parser.parse_args()

    if args.command == "baseline":
        run_baseline()
    elif args.command == "trial":
        run_trial(args.message)
    elif args.command == "loop":
        run_loop(args.n)
    elif args.command == "status":
        print_status()
    elif args.command == "profile":
        run_profiled_trial()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
