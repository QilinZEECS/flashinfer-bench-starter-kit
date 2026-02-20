"""JSONL ledger for experiment tracking."""

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import LEDGER_PATH, RUNS_DIR, COMPARISONS_DIR


class Ledger:
    """Persistent experiment ledger backed by JSONL + per-run JSON files."""

    def __init__(self, ledger_path: Path = None, runs_dir: Path = None):
        self.ledger_path = ledger_path or LEDGER_PATH
        self.runs_dir = runs_dir or RUNS_DIR
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def next_run_id(self) -> str:
        """Get the next sequential run ID (zero-padded 3 digits)."""
        existing = list(self.runs_dir.glob("run_*.json"))
        if not existing:
            return "001"
        ids = []
        for p in existing:
            try:
                ids.append(int(p.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return f"{max(ids) + 1:03d}" if ids else "001"

    def append_run(self, run_data: dict) -> str:
        """Append a run to the ledger and save full data to runs/.

        Args:
            run_data: Full run data including 'workloads' list.

        Returns:
            The run_id assigned.
        """
        run_id = run_data.get("run_id") or self.next_run_id()
        run_data["run_id"] = run_id

        if "timestamp" not in run_data:
            run_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Save full data (with workloads) to runs/
        run_path = self.runs_dir / f"run_{run_id}.json"
        run_path.write_text(json.dumps(run_data, indent=2, ensure_ascii=False))

        # Append summary (without workloads) to ledger.jsonl
        summary = {k: v for k, v in run_data.items() if k != "workloads"}
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

        return run_id

    def get_run(self, run_id: str) -> dict | None:
        """Load full run data from runs/ directory."""
        run_path = self.runs_dir / f"run_{run_id}.json"
        if not run_path.exists():
            return None
        return json.loads(run_path.read_text())

    def get_best(self) -> dict | None:
        """Get the run with highest geomean_speedup where all workloads passed."""
        runs = self.list_runs()
        passed = [r for r in runs if r.get("summary", {}).get("num_failed", 1) == 0]
        if not passed:
            return None
        return max(passed, key=lambda r: r.get("summary", {}).get("geomean_speedup", 0))

    def list_runs(self, strategy: str = None) -> list[dict]:
        """List all run summaries from the ledger, optionally filtered by strategy."""
        if not self.ledger_path.exists():
            return []
        runs = []
        for line in self.ledger_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if strategy is None or entry.get("strategy") == strategy:
                    runs.append(entry)
            except json.JSONDecodeError:
                continue
        return runs

    def save_comparison(self, comparison: dict) -> Path:
        """Save a comparison report to comparisons/."""
        comparisons_dir = COMPARISONS_DIR
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        comp_id = comparison.get("comparison_id", "unknown")
        path = comparisons_dir / f"{comp_id}.json"
        path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False))
        return path
