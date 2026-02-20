"""Benchmark result analysis, comparison, and report generation."""

import json
import hashlib
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import REPORTS_DIR, PROJECT_ROOT
from .ledger import Ledger


class Analyzer:
    """Analyze benchmark results, generate comparisons and reports."""

    def __init__(self, ledger: Ledger = None):
        self.ledger = ledger or Ledger()

    def parse_modal_results(self, results: dict, definition: str = None) -> dict:
        """Parse raw Modal benchmark results dict into structured run data.

        Args:
            results: Raw dict returned by Modal run_benchmark, e.g.
                {definition_name: {workload_uuid: {status, latency_ms, ...}}}
            definition: Definition name key. If None, uses first key.

        Returns:
            Dict with 'summary' and 'workloads' keys.
        """
        if definition is None:
            definition = next(iter(results))

        traces = results[definition]
        workloads = []
        speedups = []

        for uuid, entry in traces.items():
            wl = {
                "uuid": uuid,
                "status": entry.get("status", "UNKNOWN"),
                "speedup": entry.get("speedup_factor"),
                "latency_ms": entry.get("latency_ms"),
                "ref_latency_ms": entry.get("reference_latency_ms"),
                "max_abs_error": entry.get("max_abs_error"),
                "max_rel_error": entry.get("max_rel_error"),
            }
            workloads.append(wl)
            if wl["speedup"] is not None and wl["status"] == "PASSED":
                speedups.append(wl["speedup"])

        num_passed = sum(1 for w in workloads if w["status"] == "PASSED")
        num_failed = len(workloads) - num_passed

        if speedups:
            geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        else:
            geomean = 0.0

        summary = {
            "num_passed": num_passed,
            "num_failed": num_failed,
            "avg_speedup": round(sum(speedups) / len(speedups), 2) if speedups else 0.0,
            "min_speedup": round(min(speedups), 2) if speedups else 0.0,
            "max_speedup": round(max(speedups), 2) if speedups else 0.0,
            "geomean_speedup": round(geomean, 2),
        }

        return {"summary": summary, "workloads": workloads}

    def compare(self, run_a_id: str, run_b_id: str) -> dict:
        """Compare two runs and generate a comparison report.

        Args:
            run_a_id: Baseline run ID.
            run_b_id: Candidate run ID.

        Returns:
            Comparison dict (also saved to comparisons/).
        """
        run_a = self.ledger.get_run(run_a_id)
        run_b = self.ledger.get_run(run_b_id)

        if not run_a or not run_b:
            raise ValueError(f"Run not found: {run_a_id if not run_a else run_b_id}")

        # Build UUID→workload maps
        a_map = {w["uuid"]: w for w in run_a.get("workloads", [])}
        b_map = {w["uuid"]: w for w in run_b.get("workloads", [])}

        per_workload = []
        improved = regressed = unchanged = 0

        for uuid in set(list(a_map.keys()) + list(b_map.keys())):
            a_wl = a_map.get(uuid, {})
            b_wl = b_map.get(uuid, {})
            a_speedup = a_wl.get("speedup", 0) or 0
            b_speedup = b_wl.get("speedup", 0) or 0
            delta = round(b_speedup - a_speedup, 2)

            if delta > 0.1:
                verdict = "improved"
                improved += 1
            elif delta < -0.1:
                verdict = "regressed"
                regressed += 1
            else:
                verdict = "unchanged"
                unchanged += 1

            per_workload.append({
                "uuid": uuid,
                "baseline_speedup": round(a_speedup, 2),
                "candidate_speedup": round(b_speedup, 2),
                "delta": f"{delta:+.2f}x",
                "verdict": verdict,
            })

        comparison = {
            "comparison_id": f"cmp_{run_a_id}_vs_{run_b_id}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "baseline": {"run_id": run_a_id, "variant": run_a.get("variant", "")},
            "candidate": {"run_id": run_b_id, "variant": run_b.get("variant", "")},
            "summary": {
                "baseline_geomean": run_a.get("summary", {}).get("geomean_speedup", 0),
                "candidate_geomean": run_b.get("summary", {}).get("geomean_speedup", 0),
                "geomean_delta": f"{run_b.get('summary', {}).get('geomean_speedup', 0) - run_a.get('summary', {}).get('geomean_speedup', 0):+.2f}x",
                "improved_workloads": improved,
                "regressed_workloads": regressed,
                "unchanged_workloads": unchanged,
            },
            "per_workload": sorted(per_workload, key=lambda w: float(w["delta"].rstrip("x")), reverse=True),
        }

        self.ledger.save_comparison(comparison)
        return comparison

    def identify_bottleneck(self, run_id: str) -> str:
        """Identify the weakest workloads in a run, grouped by rough T size."""
        run = self.ledger.get_run(run_id)
        if not run:
            return f"Run {run_id} not found."

        workloads = sorted(run.get("workloads", []), key=lambda w: w.get("speedup", 0) or 0)
        lines = [f"Bottleneck analysis for Run {run_id}:", ""]

        # Show worst 5
        lines.append("Worst 5 workloads:")
        for w in workloads[:5]:
            lines.append(f"  {w['uuid'][:8]}... speedup={w.get('speedup', 'N/A')}x  latency={w.get('latency_ms', 'N/A')}ms")

        return "\n".join(lines)

    def generate_report(
        self,
        run_id: str,
        strategy: str = "",
        description: str = "",
        code_diff: str = "",
        baseline_run_id: str = None,
    ) -> Path:
        """Generate a full Markdown experiment report.

        Args:
            run_id: The run to generate a report for.
            strategy: Strategy label (e.g., "P0", "baseline").
            description: Description of the optimization approach.
            code_diff: Git diff of kernel changes (optional).
            baseline_run_id: Run to compare against (optional).

        Returns:
            Path to the generated report.
        """
        run = self.ledger.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found.")

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        variant_name = run.get("variant", "unknown").replace(" ", "_")
        report_path = REPORTS_DIR / f"run_{run_id}_{variant_name}.md"

        summary = run.get("summary", {})
        workloads = run.get("workloads", [])

        lines = []
        lines.append(f"# Run {run_id}: {run.get('variant', 'Unknown')} ({strategy})")
        lines.append("")

        # Basic info
        lines.append("## 基本信息")
        lines.append(f"- **Run ID**: {run_id}")
        lines.append(f"- **时间**: {run.get('timestamp', 'N/A')}")
        lines.append(f"- **策略**: {strategy}")
        lines.append(f"- **分支**: {run.get('git_branch', 'N/A')}")
        lines.append(f"- **Commit**: {run.get('git_commit', 'N/A')}")
        lines.append(f"- **Kernel SHA256**: {run.get('kernel_sha256', 'N/A')}")
        lines.append("")

        # Optimization approach
        lines.append("## 优化思路")
        if description:
            lines.append(description)
        else:
            lines.append("(baseline — 无优化)")
        lines.append("")

        # Code changes
        if code_diff:
            lines.append("## 关键代码变更")
            lines.append("```diff")
            lines.append(code_diff)
            lines.append("```")
            lines.append("")

        # Results overview
        lines.append("## 结果总览")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| Passed / Total | {summary.get('num_passed', 0)} / {summary.get('num_passed', 0) + summary.get('num_failed', 0)} |")
        lines.append(f"| Avg Speedup | {summary.get('avg_speedup', 0)}x |")
        lines.append(f"| Geomean Speedup | {summary.get('geomean_speedup', 0)}x |")

        # Find min/max workloads
        passed_wl = [w for w in workloads if w.get("status") == "PASSED" and w.get("speedup")]
        if passed_wl:
            min_wl = min(passed_wl, key=lambda w: w["speedup"])
            max_wl = max(passed_wl, key=lambda w: w["speedup"])
            lines.append(f"| Min Speedup | {min_wl['speedup']}x ({min_wl['uuid'][:8]}...) |")
            lines.append(f"| Max Speedup | {max_wl['speedup']}x ({max_wl['uuid'][:8]}...) |")
        lines.append("")

        # Per-workload table
        lines.append("## Per-Workload 结果")
        lines.append("| # | UUID (短) | Speedup | Latency (ms) | Ref Latency (ms) | 状态 |")
        lines.append("|---|-----------|---------|-------------|-----------------|------|")
        for i, w in enumerate(sorted(workloads, key=lambda x: x.get("speedup", 0) or 0), 1):
            lines.append(
                f"| {i} "
                f"| {w['uuid'][:8]}... "
                f"| {w.get('speedup', 'N/A')}x "
                f"| {w.get('latency_ms', 'N/A')} "
                f"| {w.get('ref_latency_ms', 'N/A')} "
                f"| {w.get('status', 'UNKNOWN')} |"
            )
        lines.append("")

        # Comparison with baseline
        if baseline_run_id:
            baseline = self.ledger.get_run(baseline_run_id)
            if baseline:
                b_summary = baseline.get("summary", {})
                lines.append(f"## 与基线的对比 (vs Run {baseline_run_id})")
                lines.append("| 指标 | Baseline | This Run | Delta |")
                lines.append("|------|----------|----------|-------|")
                b_geo = b_summary.get("geomean_speedup", 0)
                c_geo = summary.get("geomean_speedup", 0)
                lines.append(f"| Geomean | {b_geo}x | {c_geo}x | {c_geo - b_geo:+.2f}x |")
                lines.append("")

        # Analysis placeholder (to be filled by Claude in conversation)
        lines.append("## 分析")
        lines.append(description if description else "(自动生成的基线记录)")
        lines.append("")

        lines.append("## 结论与下一步")
        lines.append("(待分析)")
        lines.append("")

        report_path.write_text("\n".join(lines))
        return report_path

    @staticmethod
    def kernel_sha256(kernel_path: Path) -> str:
        """Compute SHA256 hash of a kernel file."""
        content = kernel_path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]

    @staticmethod
    def get_git_info() -> dict:
        """Get current git branch and commit hash."""
        try:
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            ).stdout.strip()
            commit = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            ).stdout.strip()
            return {"git_branch": branch, "git_commit": commit}
        except Exception:
            return {"git_branch": "unknown", "git_commit": "unknown"}
