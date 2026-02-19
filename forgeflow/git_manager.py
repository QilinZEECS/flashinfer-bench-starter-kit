"""Git branching and version control for ForgeFlow trials."""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from forgeflow.config import (
    MAIN_BRANCH,
    MIN_SPEEDUP_IMPROVEMENT_PCT,
    PROJECT_ROOT,
    TRIAL_BRANCH_PREFIX,
)


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def ensure_repo_initialized() -> None:
    """Initialize git repo if not already done."""
    git_dir = PROJECT_ROOT / ".git"
    if not git_dir.exists():
        _run_git("init")
        _run_git("add", "-A")
        _run_git("commit", "-m", "Initial commit")


def get_current_branch() -> str:
    return _run_git("rev-parse", "--abbrev-ref", "HEAD")


def get_current_commit_sha() -> str:
    return _run_git("rev-parse", "--short", "HEAD")


def get_next_trial_id() -> int:
    """Find the next trial number from existing branches."""
    try:
        branches = _run_git("branch", "--list", f"{TRIAL_BRANCH_PREFIX}*")
    except RuntimeError:
        return 1

    if not branches:
        return 1

    max_id = 0
    for line in branches.splitlines():
        branch_name = line.strip().lstrip("* ")
        if not branch_name.startswith(TRIAL_BRANCH_PREFIX):
            continue
        suffix = branch_name[len(TRIAL_BRANCH_PREFIX):]
        try:
            trial_num = int(suffix)
            max_id = max(max_id, trial_num)
        except (ValueError, IndexError):
            continue
    return max_id + 1


def create_trial_branch(trial_id: int) -> str:
    """Create and checkout a new trial branch from main."""
    branch_name = f"{TRIAL_BRANCH_PREFIX}{trial_id}"
    current = get_current_branch()

    if current != MAIN_BRANCH:
        _run_git("checkout", MAIN_BRANCH)

    _run_git("checkout", "-b", branch_name)
    return branch_name


def commit_trial(trial_id: int, change_summary: str, metrics: dict) -> str:
    """Stage and commit kernel changes with performance metrics."""
    _run_git("add", "solution/triton/kernel.py")
    _run_git("add", "experiments/")
    _run_git("add", "reports/")

    avg_speedup = metrics.get("avg_speedup", 0)
    min_speedup = metrics.get("min_speedup", 0)
    passed = metrics.get("workloads_passed", 0)
    total = metrics.get("workloads_total", 0)

    message = (
        f"opt(trial-{trial_id}): {change_summary}\n\n"
        f"Performance:\n"
        f"  avg_speedup: {avg_speedup:.2f}x\n"
        f"  min_speedup: {min_speedup:.2f}x\n"
        f"  correctness: {passed}/{total} PASSED\n"
        f"  timestamp: {datetime.now(timezone.utc).isoformat()}"
    )

    _run_git("commit", "-m", message)
    return get_current_commit_sha()


def merge_trial_to_main(trial_id: int) -> None:
    """Merge current trial branch into main."""
    branch_name = f"{TRIAL_BRANCH_PREFIX}{trial_id}"
    _run_git("checkout", MAIN_BRANCH)
    _run_git("merge", branch_name, "--no-ff", "-m",
             f"Merge {branch_name}: performance improvement > {MIN_SPEEDUP_IMPROVEMENT_PCT}%")


def rollback_trial() -> None:
    """Discard uncommitted changes and return to main."""
    try:
        _run_git("checkout", "--", "solution/triton/kernel.py")
    except RuntimeError:
        pass  # File may not have changes
    try:
        _run_git("clean", "-fd", "experiments/", "reports/")
    except RuntimeError:
        pass  # Directories may not exist or have no untracked files
    current = get_current_branch()
    if current != MAIN_BRANCH:
        _run_git("checkout", MAIN_BRANCH)


def should_merge(baseline_avg_speedup: float, new_avg_speedup: float) -> bool:
    """Determine if the improvement warrants merging to main."""
    if baseline_avg_speedup <= 0:
        return new_avg_speedup > 0
    improvement_pct = ((new_avg_speedup - baseline_avg_speedup) / baseline_avg_speedup) * 100
    return improvement_pct > MIN_SPEEDUP_IMPROVEMENT_PCT


def has_uncommitted_changes() -> bool:
    """Check if there are uncommitted changes to tracked trial files."""
    try:
        diff_staged = _run_git("diff", "--cached", "--name-only")
        diff_unstaged = _run_git("diff", "--name-only")
        return bool(diff_staged.strip() or diff_unstaged.strip())
    except RuntimeError:
        return False
