"""Core runner: swap kernel → pack → modal benchmark → collect results."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from .config import (
    KERNEL_PATH, KERNEL_BACKUP_PATH, PROJECT_ROOT,
    QUICK_BENCH, FULL_BENCH,
)
from .analyzer import Analyzer
from .ledger import Ledger


class Runner:
    """Execute a kernel variant benchmark on Modal B200."""

    def __init__(self, ledger: Ledger = None, analyzer: Analyzer = None):
        self.ledger = ledger or Ledger()
        self.analyzer = analyzer or Analyzer(self.ledger)

    def run_variant(
        self,
        variant_path: Path = None,
        variant_name: str = "baseline",
        strategy: str = "baseline",
        description: str = "",
        quick: bool = False,
    ) -> dict:
        """Run a kernel variant through the full benchmark pipeline.

        Args:
            variant_path: Path to the variant kernel.py. If None, uses current kernel.
            variant_name: Human-readable name for this variant.
            strategy: Strategy label (e.g., "P0", "P1").
            description: Description of the optimization approach.
            quick: Use quick benchmark config (cheaper, less accurate).

        Returns:
            Full run data dict with summary and workloads.
        """
        try:
            # 1. Backup current kernel
            if variant_path and variant_path != KERNEL_PATH:
                self._backup_kernel()
                shutil.copy2(variant_path, KERNEL_PATH)

            # 2. Pack solution
            print(f"Packing solution...")
            self._pack_solution()

            # 3. Run Modal benchmark
            bench_config = QUICK_BENCH if quick else FULL_BENCH
            print(f"Running Modal benchmark ({'quick' if quick else 'full'} mode)...")
            results = self._run_modal(bench_config)

            if not results:
                raise RuntimeError("Modal benchmark returned no results")

            # 4. Parse results
            parsed = self.analyzer.parse_modal_results(results)

            # 5. Build run data
            git_info = Analyzer.get_git_info()
            run_id = self.ledger.next_run_id()

            run_data = {
                "run_id": run_id,
                "variant": variant_name,
                "strategy": strategy,
                "kernel_sha256": Analyzer.kernel_sha256(KERNEL_PATH),
                "bench_config": bench_config,
                **git_info,
                **parsed,
            }

            # 6. Save to ledger
            self.ledger.append_run(run_data)

            # 7. Generate report
            best = self.ledger.get_best()
            baseline_id = None
            if best and best.get("run_id") != run_id:
                baseline_id = best["run_id"]

            code_diff = ""
            if variant_path and variant_path != KERNEL_PATH:
                code_diff = self._get_code_diff()

            report_path = self.analyzer.generate_report(
                run_id=run_id,
                strategy=strategy,
                description=description,
                code_diff=code_diff,
                baseline_run_id=baseline_id,
            )

            print(f"\nRun {run_id} complete:")
            summary = run_data["summary"]
            print(f"  Passed: {summary['num_passed']}/{summary['num_passed'] + summary['num_failed']}")
            print(f"  Avg speedup: {summary['avg_speedup']}x")
            print(f"  Geomean speedup: {summary['geomean_speedup']}x")
            print(f"  Min: {summary['min_speedup']}x  Max: {summary['max_speedup']}x")
            print(f"  Report: {report_path}")

            return run_data

        finally:
            # Always restore backup
            self._restore_kernel()

    def run_current(self, quick: bool = False) -> dict:
        """Run benchmark on the current kernel without swapping."""
        return self.run_variant(
            variant_path=None,
            variant_name="current",
            strategy="current",
            description="Benchmark of the current kernel.py in place.",
            quick=quick,
        )

    def _backup_kernel(self):
        """Backup the current kernel file."""
        if KERNEL_PATH.exists():
            shutil.copy2(KERNEL_PATH, KERNEL_BACKUP_PATH)

    def _restore_kernel(self):
        """Restore kernel from backup if backup exists."""
        if KERNEL_BACKUP_PATH.exists():
            shutil.copy2(KERNEL_BACKUP_PATH, KERNEL_PATH)
            KERNEL_BACKUP_PATH.unlink()

    def _pack_solution(self):
        """Run pack_solution.py to create solution.json."""
        result = subprocess.run(
            ["conda", "run", "-n", "fi-bench",
             "python", str(PROJECT_ROOT / "scripts" / "pack_solution.py")],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            raise RuntimeError(f"pack_solution failed:\n{result.stderr}")

    def _run_modal(self, bench_config: dict) -> dict:
        """Run Modal benchmark and capture results as dict.

        Uses modal run to execute run_modal.py with JSON output.
        """
        result = subprocess.run(
            ["conda", "run", "-n", "fi-bench",
             "python", "-m", "modal", "run", str(PROJECT_ROOT / "scripts" / "run_modal.py")],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            timeout=1800,
        )

        if result.returncode != 0:
            print(f"Modal stderr:\n{result.stderr}", file=sys.stderr)
            raise RuntimeError(f"Modal run failed with code {result.returncode}")

        # Parse the output to extract results
        return self._parse_modal_output(result.stdout)

    def _parse_modal_output(self, output: str) -> dict:
        """Parse Modal stdout to extract benchmark results.

        The Modal script prints results in a structured format. We parse
        the workload lines to reconstruct the results dict.
        """
        definition = None
        workloads = {}

        for line in output.strip().split("\n"):
            line = line.strip()

            # Detect definition header
            if line.endswith(":") and not line.startswith("Workload") and not line.startswith("Error"):
                potential_def = line.rstrip(":")
                if "moe_fp8" in potential_def:
                    definition = potential_def
                continue

            # Parse workload result lines
            if line.startswith("Workload "):
                entry = self._parse_workload_line(line)
                if entry:
                    workloads[entry["uuid"]] = entry

        if not definition:
            definition = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"

        return {definition: workloads}

    @staticmethod
    def _parse_workload_line(line: str) -> dict | None:
        """Parse a single workload result line from Modal output.

        Example line:
          Workload a1b2c3d4...: PASSED | 1.234 ms | 5.67x speedup | abs_err=1.2e-03, rel_err=2.1e-04
        """
        try:
            # Extract UUID
            uuid_part = line.split("Workload ")[1].split("...")[0]
            # We only have abbreviated UUID from output; use what we have
            uuid = uuid_part

            parts = line.split("|")
            status_part = parts[0].split(":")[-1].strip()

            entry = {"uuid": uuid, "status": status_part}

            for part in parts[1:]:
                part = part.strip()
                if "ms" in part and "speedup" not in part:
                    entry["latency_ms"] = float(part.replace("ms", "").strip())
                elif "speedup" in part:
                    entry["speedup_factor"] = float(part.replace("x speedup", "").strip())
                elif "abs_err" in part:
                    for kv in part.split(","):
                        kv = kv.strip()
                        if "abs_err" in kv:
                            entry["max_abs_error"] = float(kv.split("=")[1])
                        elif "rel_err" in kv:
                            entry["max_rel_error"] = float(kv.split("=")[1])

            return entry
        except (IndexError, ValueError):
            return None

    def _get_code_diff(self) -> str:
        """Get git diff of kernel.py vs main branch."""
        try:
            result = subprocess.run(
                ["git", "diff", "main", "--", "solution/triton/kernel.py"],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            )
            return result.stdout[:3000] if result.stdout else ""
        except Exception:
            return ""
