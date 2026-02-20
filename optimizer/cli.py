"""CLI entry point: python -m optimizer <command>."""

import argparse
import json
import sys
from pathlib import Path

from .config import KERNEL_PATH
from .ledger import Ledger
from .analyzer import Analyzer
from .runner import Runner
from .sweep import SweepConfig, run_sweep


def cmd_run(args):
    """Run benchmark on a kernel variant."""
    ledger = Ledger()
    runner = Runner(ledger)

    if args.variant:
        variant_path = Path(args.variant).resolve()
        if not variant_path.exists():
            print(f"Error: variant file not found: {variant_path}", file=sys.stderr)
            sys.exit(1)
        run_data = runner.run_variant(
            variant_path=variant_path,
            variant_name=variant_path.stem,
            strategy=args.strategy or "manual",
            description=args.description or "",
            quick=args.quick,
        )
    else:
        run_data = runner.run_variant(
            variant_path=None,
            variant_name=args.name or "baseline",
            strategy=args.strategy or "baseline",
            description=args.description or "Baseline benchmark run.",
            quick=args.quick,
        )

    return run_data


def cmd_status(args):
    """Show ledger summary table."""
    ledger = Ledger()
    runs = ledger.list_runs(strategy=args.strategy)

    if not runs:
        print("No runs recorded yet.")
        return

    # Header
    print(f"{'ID':>4} {'Variant':<30} {'Strategy':<10} {'Pass':>5} {'Fail':>5} "
          f"{'Avg':>7} {'Geo':>7} {'Min':>7} {'Max':>7}")
    print("-" * 95)

    for r in runs:
        s = r.get("summary", {})
        print(f"{r.get('run_id', '?'):>4} "
              f"{r.get('variant', 'N/A'):<30} "
              f"{r.get('strategy', 'N/A'):<10} "
              f"{s.get('num_passed', 0):>5} "
              f"{s.get('num_failed', 0):>5} "
              f"{s.get('avg_speedup', 0):>6.2f}x "
              f"{s.get('geomean_speedup', 0):>6.2f}x "
              f"{s.get('min_speedup', 0):>6.2f}x "
              f"{s.get('max_speedup', 0):>6.2f}x")


def cmd_compare(args):
    """Compare two runs."""
    ledger = Ledger()
    analyzer = Analyzer(ledger)

    comparison = analyzer.compare(args.run_a, args.run_b)

    s = comparison["summary"]
    print(f"Comparison: Run {args.run_a} vs Run {args.run_b}")
    print(f"  Baseline geomean:  {s['baseline_geomean']}x")
    print(f"  Candidate geomean: {s['candidate_geomean']}x")
    print(f"  Delta:             {s['geomean_delta']}")
    print(f"  Improved:  {s['improved_workloads']}")
    print(f"  Regressed: {s['regressed_workloads']}")
    print(f"  Unchanged: {s['unchanged_workloads']}")

    if args.detail:
        print(f"\nPer-workload:")
        for w in comparison["per_workload"]:
            print(f"  {w['uuid'][:8]}... {w['baseline_speedup']}x → {w['candidate_speedup']}x ({w['delta']}) [{w['verdict']}]")


def cmd_best(args):
    """Show the best run."""
    ledger = Ledger()
    best = ledger.get_best()

    if not best:
        print("No passing runs recorded yet.")
        return

    s = best.get("summary", {})
    print(f"Best run: {best.get('run_id')}")
    print(f"  Variant:  {best.get('variant')}")
    print(f"  Strategy: {best.get('strategy')}")
    print(f"  Branch:   {best.get('git_branch')}")
    print(f"  Geomean:  {s.get('geomean_speedup')}x")
    print(f"  Avg:      {s.get('avg_speedup')}x")
    print(f"  Min/Max:  {s.get('min_speedup')}x / {s.get('max_speedup')}x")
    print(f"  Passed:   {s.get('num_passed')}/{s.get('num_passed', 0) + s.get('num_failed', 0)}")


def cmd_install(args):
    """Install a run's kernel as the active kernel."""
    ledger = Ledger()
    run = ledger.get_run(args.run_id)

    if not run:
        print(f"Run {args.run_id} not found.", file=sys.stderr)
        sys.exit(1)

    # Check if run has a git branch we can checkout from
    branch = run.get("git_branch")
    if branch and branch != "main":
        import subprocess
        from .config import PROJECT_ROOT
        result = subprocess.run(
            ["git", "checkout", branch, "--", "solution/triton/kernel.py"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            print(f"Installed kernel from branch '{branch}' (run {args.run_id})")
            return

    print(f"Cannot install: run {args.run_id} has no recoverable kernel source.", file=sys.stderr)
    sys.exit(1)


def cmd_sweep(args):
    """Run a parameter sweep."""
    params = {}
    for param_str in args.param:
        name, values_str = param_str.split("=", 1)
        values = []
        for v in values_str.split(","):
            v = v.strip()
            # Try int, then float, then string
            try:
                values.append(int(v))
            except ValueError:
                try:
                    values.append(float(v))
                except ValueError:
                    # Handle bool-like
                    if v.lower() in ("true", "false"):
                        values.append(v.lower() == "true")
                    else:
                        values.append(v)
        params[name] = values

    config = SweepConfig(
        params=params,
        base_kernel=Path(args.base).resolve() if args.base else None,
        quick=not args.full,
        strategy=args.strategy or "sweep",
    )

    print(f"Sweep config: {config.num_variants} variants")
    for k, v in config.params.items():
        print(f"  {k}: {v}")

    run_sweep(config)


def cmd_report(args):
    """Regenerate report for a run."""
    ledger = Ledger()
    analyzer = Analyzer(ledger)

    report_path = analyzer.generate_report(
        run_id=args.run_id,
        strategy=args.strategy or "",
        description=args.description or "",
        baseline_run_id=args.baseline,
    )
    print(f"Report generated: {report_path}")


def main():
    parser = argparse.ArgumentParser(prog="optimizer", description="Kernel optimization system")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = subparsers.add_parser("run", help="Run benchmark on a kernel variant")
    p_run.add_argument("--variant", "-v", help="Path to variant kernel.py")
    p_run.add_argument("--name", "-n", help="Variant name")
    p_run.add_argument("--strategy", "-s", help="Strategy label (e.g., P0, P1)")
    p_run.add_argument("--description", "-d", help="Description of the optimization")
    p_run.add_argument("--quick", "-q", action="store_true", help="Quick benchmark mode")
    p_run.set_defaults(func=cmd_run)

    # status
    p_status = subparsers.add_parser("status", help="Show ledger summary")
    p_status.add_argument("--strategy", "-s", help="Filter by strategy")
    p_status.set_defaults(func=cmd_status)

    # compare
    p_compare = subparsers.add_parser("compare", help="Compare two runs")
    p_compare.add_argument("run_a", help="Baseline run ID")
    p_compare.add_argument("run_b", help="Candidate run ID")
    p_compare.add_argument("--detail", "-d", action="store_true", help="Show per-workload detail")
    p_compare.set_defaults(func=cmd_compare)

    # best
    p_best = subparsers.add_parser("best", help="Show best run")
    p_best.set_defaults(func=cmd_best)

    # install
    p_install = subparsers.add_parser("install", help="Install a run's kernel")
    p_install.add_argument("run_id", help="Run ID to install")
    p_install.set_defaults(func=cmd_install)

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Run parameter sweep")
    p_sweep.add_argument("--param", "-p", action="append", required=True,
                         help="Parameter: NAME=val1,val2,val3")
    p_sweep.add_argument("--base", "-b", help="Base kernel template path")
    p_sweep.add_argument("--strategy", "-s", help="Strategy label")
    p_sweep.add_argument("--full", action="store_true", help="Use full benchmark mode")
    p_sweep.set_defaults(func=cmd_sweep)

    # report
    p_report = subparsers.add_parser("report", help="Regenerate report for a run")
    p_report.add_argument("run_id", help="Run ID")
    p_report.add_argument("--strategy", "-s", help="Strategy label")
    p_report.add_argument("--description", "-d", help="Optimization description")
    p_report.add_argument("--baseline", help="Baseline run ID for comparison")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
