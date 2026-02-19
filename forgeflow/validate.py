"""Correctness validation via flashinfer-bench on Modal B200."""

import json
import subprocess
import sys
from pathlib import Path

from forgeflow.config import DEFINITION_NAME, PROJECT_ROOT, SOLUTION_JSON


def pack_solution() -> Path:
    """Run pack_solution.py and return path to solution.json."""
    result = subprocess.run(
        [sys.executable, "scripts/pack_solution.py"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pack_solution failed: {result.stderr}")
    if not SOLUTION_JSON.exists():
        raise FileNotFoundError(f"solution.json not created at {SOLUTION_JSON}")
    return SOLUTION_JSON


def run_modal_benchmark() -> dict:
    """Run benchmark on Modal B200 and capture structured results.

    Returns dict keyed by workload_uuid with status, latency, speedup, errors.
    """
    result = subprocess.run(
        ["modal", "run", "scripts/run_modal.py"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )

    output = result.stdout + result.stderr
    return _parse_benchmark_output(output)


def _parse_benchmark_output(output: str) -> dict:
    """Parse benchmark CLI output into structured results."""
    results = {}
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("Workload "):
            continue

        parts = line.split("|")
        if len(parts) < 2:
            continue

        header = parts[0].strip()
        workload_id = header.split("...")[0].replace("Workload ", "").strip()
        status = header.split(":")[-1].strip()

        entry = {"status": status, "workload_id": workload_id}

        for part in parts[1:]:
            part = part.strip()
            if part.endswith("ms"):
                entry["latency_ms"] = float(part.replace("ms", "").strip())
            elif part.endswith("speedup"):
                entry["speedup"] = float(part.replace("x speedup", "").strip())
            elif part.startswith("abs_err="):
                kv_pairs = part.split(",")
                for kv in kv_pairs:
                    kv = kv.strip()
                    if kv.startswith("abs_err="):
                        entry["abs_err"] = float(kv.split("=")[1])
                    elif kv.startswith("rel_err="):
                        entry["rel_err"] = float(kv.split("=")[1])

        results[workload_id] = entry

    return results


def validate_correctness(results: dict) -> tuple[bool, int, int]:
    """Check that all workloads have PASSED status.

    Returns (all_passed, num_passed, num_total).
    """
    total = len(results)
    passed = sum(1 for r in results.values() if r.get("status") == "PASSED")
    return passed == total, passed, total


def extract_metrics(results: dict) -> dict:
    """Extract aggregate performance metrics from benchmark results."""
    speedups = [r["speedup"] for r in results.values() if "speedup" in r]
    latencies = [r["latency_ms"] for r in results.values() if "latency_ms" in r]

    if not speedups:
        return {
            "avg_speedup": 0.0,
            "min_speedup": 0.0,
            "max_speedup": 0.0,
            "avg_latency_ms": 0.0,
            "workloads_passed": 0,
            "workloads_total": len(results),
        }

    _, num_passed, num_total = validate_correctness(results)

    return {
        "avg_speedup": sum(speedups) / len(speedups),
        "min_speedup": min(speedups),
        "max_speedup": max(speedups),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "workloads_passed": num_passed,
        "workloads_total": num_total,
    }


def run_full_validation() -> tuple[bool, dict, dict]:
    """Pack, benchmark, and validate in one call.

    Returns (all_passed, raw_results, aggregate_metrics).
    """
    print("[validate] Packing solution...")
    pack_solution()

    print("[validate] Running Modal B200 benchmark...")
    results = run_modal_benchmark()

    if not results:
        raise RuntimeError("No benchmark results returned. Check Modal connectivity.")

    all_passed, num_passed, num_total = validate_correctness(results)
    metrics = extract_metrics(results)

    status_str = "PASS" if all_passed else "FAIL"
    print(f"[validate] {status_str}: {num_passed}/{num_total} workloads passed")
    print(f"[validate] avg_speedup={metrics['avg_speedup']:.2f}x  "
          f"min={metrics['min_speedup']:.2f}x  max={metrics['max_speedup']:.2f}x")

    return all_passed, results, metrics
