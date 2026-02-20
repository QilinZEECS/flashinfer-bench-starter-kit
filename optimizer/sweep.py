"""System-level parameter sweep: template-based kernel generation + benchmark."""

import itertools
import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .config import KERNEL_PATH, SWEEPS_DIR
from .ledger import Ledger
from .runner import Runner


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep."""
    params: dict[str, list]   # {"GROUP_SIZE": [1,2,4], "SPLIT_K": [1,2]}
    base_kernel: Path = None  # Template kernel path (default: current kernel)
    quick: bool = True        # Use quick benchmark mode
    strategy: str = "sweep"

    @property
    def combinations(self) -> list[dict]:
        """Generate all parameter combinations (Cartesian product)."""
        keys = list(self.params.keys())
        values = list(self.params.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    @property
    def num_variants(self) -> int:
        return len(self.combinations)


def generate_variant(base_kernel_path: Path, params: dict, output_path: Path) -> Path:
    """Generate a kernel variant by substituting template variables.

    Looks for lines like:
        SWEEP_GROUP_SIZE = 1
    and replaces the value with the sweep parameter.

    Args:
        base_kernel_path: Template kernel file.
        params: Dict of param_name → value to substitute.
        output_path: Where to write the variant.

    Returns:
        Path to the generated variant.
    """
    content = base_kernel_path.read_text()

    for param_name, value in params.items():
        # Match SWEEP_<PARAM> = <value> pattern
        sweep_var = f"SWEEP_{param_name}"
        pattern = rf"^({sweep_var}\s*=\s*).*$"
        replacement = rf"\g<1>{repr(value)}"
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    output_path.write_text(content)
    return output_path


def run_sweep(config: SweepConfig) -> dict:
    """Run a full parameter sweep.

    Generates all variant kernels, benchmarks each one, and returns
    aggregated results with the best combination.

    Args:
        config: Sweep configuration.

    Returns:
        Sweep result dict (also saved to results/sweeps/).
    """
    ledger = Ledger()
    runner = Runner(ledger)
    base_kernel = config.base_kernel or KERNEL_PATH

    combinations = config.combinations
    print(f"Starting sweep: {config.num_variants} variants")
    print(f"Parameters: {config.params}")
    print()

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, combo in enumerate(combinations, 1):
            combo_str = ", ".join(f"{k}={v}" for k, v in combo.items())
            print(f"[{i}/{len(combinations)}] {combo_str}")

            variant_path = Path(tmpdir) / f"variant_{i:03d}.py"
            generate_variant(base_kernel, combo, variant_path)

            try:
                run_data = runner.run_variant(
                    variant_path=variant_path,
                    variant_name=f"sweep_{combo_str}",
                    strategy=config.strategy,
                    description=f"Sweep variant: {combo_str}",
                    quick=config.quick,
                )
                results.append({
                    "params": combo,
                    "run_id": run_data["run_id"],
                    "geomean": run_data["summary"]["geomean_speedup"],
                    "avg": run_data["summary"]["avg_speedup"],
                    "passed": run_data["summary"]["num_passed"],
                    "failed": run_data["summary"]["num_failed"],
                })
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({
                    "params": combo,
                    "run_id": None,
                    "geomean": 0,
                    "error": str(e),
                })

    # Find best (all passed, highest geomean)
    valid = [r for r in results if r.get("failed", 1) == 0 and r.get("run_id")]
    best = max(valid, key=lambda r: r["geomean"]) if valid else None

    # Build sweep summary
    sweep_id = _next_sweep_id()
    sweep_data = {
        "sweep_id": sweep_id,
        "params": config.params,
        "num_variants": len(combinations),
        "strategy": config.strategy,
        "quick": config.quick,
        "results": results,
        "best": best,
    }

    # Save
    SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
    sweep_path = SWEEPS_DIR / f"sweep_{sweep_id}.json"
    sweep_path.write_text(json.dumps(sweep_data, indent=2, ensure_ascii=False))

    # Print summary
    print(f"\nSweep {sweep_id} complete: {len(results)} variants tested")
    if best:
        print(f"Best: {best['params']} → geomean {best['geomean']}x (run {best['run_id']})")
    else:
        print("No valid results found.")
    print(f"Saved: {sweep_path}")

    return sweep_data


def _next_sweep_id() -> str:
    """Get next sequential sweep ID."""
    SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(SWEEPS_DIR.glob("sweep_*.json"))
    if not existing:
        return "001"
    ids = []
    for p in existing:
        try:
            ids.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return f"{max(ids) + 1:03d}" if ids else "001"
