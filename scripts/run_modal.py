"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Packs the solution from source files and runs benchmarks on NVIDIA B200 GPUs via
Modal. Local machine only needs `modal`; flashinfer-bench dependencies are loaded
inside the remote Modal container.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
try:
    import tomllib
except ImportError:
    import tomli as tomllib

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"
TRACE_SET_DATA_PATH = "/data/mlsys26-contest"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


def load_config() -> dict:
    """Load benchmark configuration from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def collect_solution_files(language: str) -> dict:
    """Collect local solution source files for remote packing."""
    if language == "triton":
        source_dir = PROJECT_ROOT / "solution" / "triton"
    elif language == "cuda":
        source_dir = PROJECT_ROOT / "solution" / "cuda"
    else:
        raise ValueError(f"Unsupported language: {language}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    files = {}
    for file_path in source_dir.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(source_dir).as_posix()
            files[rel_path] = file_path.read_text()

    if not files:
        raise ValueError(f"No source files found in {source_dir}")

    return files


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(config_data: dict, source_files: dict, benchmark_config: dict = None) -> dict:
    """Run benchmark on Modal B200 and return results."""
    from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, TraceSet
    from flashinfer_bench.agents import pack_solution_from_files

    solution_config = config_data["solution"]
    build_config = config_data["build"]

    if benchmark_config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
    else:
        config = BenchmarkConfig(**benchmark_config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        source_dir = Path(tmp_dir) / "solution_src"
        source_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, file_content in source_files.items():
            dst = source_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(file_content)

        spec = BuildSpec(
            language=build_config["language"],
            target_hardware=["cuda"],
            entry_point=build_config["entry_point"],
        )
        solution = pack_solution_from_files(
            path=str(source_dir),
            spec=spec,
            name=solution_config["name"],
            definition=solution_config["definition"],
            author=solution_config["author"],
        )

    trace_set = TraceSet.from_path(TRACE_SET_DATA_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


@app.local_entrypoint()
def main():
    """Pack solution on Modal and run benchmark on Modal B200."""
    print("Loading local config and source files...")
    config = load_config()
    source_files = collect_solution_files(config["build"]["language"])
    print(
        f"Loaded {len(source_files)} source file(s) for "
        f"{config['solution']['name']} ({config['solution']['definition']})."
    )

    print("\nRunning pack + benchmark on Modal B200...")
    results = run_benchmark.remote(config, source_files)

    if not results:
        print("No results returned!")
        return

    print_results(results)
