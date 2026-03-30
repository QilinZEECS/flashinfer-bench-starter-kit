"""
FlashInfer-Bench Modal Cloud Benchmark Runner — Single workload validation.

Same as run_modal.py but only runs the first workload for quick iteration.
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

from scripts.torch_builder_flags import (
    build_name_with_cuda_flags,
    maybe_patch_torch_cpp_extension,
)

app = modal.App("flashinfer-bench-validate")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"
TRACE_SET_DATA_PATH = "/data/mlsys26-contest"
CUTLASS_INCLUDE = "/opt/cutlass/include"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .add_local_python_source("scripts", copy=True)
    .run_commands(
        "apt-get update && apt-get install -y git && "
        "git clone --depth 1 https://github.com/NVIDIA/cutlass.git /opt/cutlass"
    )
    .pip_install("flashinfer-bench", "torch==2.8.0", "triton", "numpy")
)


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def collect_solution_files(language: str) -> dict:
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
    import os
    from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, TraceSet
    from flashinfer_bench.agents import pack_solution_from_files

    os.environ["CPATH"] = (
        CUTLASS_INCLUDE
        if not os.environ.get("CPATH")
        else f"{CUTLASS_INCLUDE}:{os.environ['CPATH']}"
    )

    solution_config = config_data["solution"]
    build_config = config_data["build"]
    extra_cuda_cflags = maybe_patch_torch_cpp_extension(build_config)

    if benchmark_config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
    else:
        config = BenchmarkConfig(**benchmark_config)

    if extra_cuda_cflags:
        print(f"Injecting extra CUDA flags: {extra_cuda_cflags}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        source_dir = Path(tmp_dir) / "solution_src"
        source_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, file_content in source_files.items():
            dst = source_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(file_content)

        spec_kwargs = {
            "language": build_config["language"],
            "target_hardware": ["cuda"],
            "entry_point": build_config["entry_point"],
        }
        for key in ("dependencies", "binding", "destination_passing_style"):
            if key in build_config:
                spec_kwargs[key] = build_config[key]
        spec = BuildSpec(**spec_kwargs)
        solution = pack_solution_from_files(
            path=str(source_dir),
            spec=spec,
            name=build_name_with_cuda_flags(solution_config["name"], build_config),
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

    # Only run first workload for quick validation
    workloads = workloads[:1]

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
            if trace.evaluation.log:
                entry["log"] = trace.evaluation.log
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
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

            if result.get("status") not in {"PASSED"} and result.get("log"):
                log = result["log"]
                lines = log.splitlines()
                err_lines = [
                    ln for ln in lines
                    if (" error:" in ln.lower())
                    or ("fatal error" in ln.lower())
                    or ("undefined reference" in ln.lower())
                    or ("ninja: build stopped" in ln.lower())
                    or ("runtimeerror: error building extension" in ln.lower())
                ]
                if err_lines:
                    snippet = "\n".join(err_lines[-20:])
                else:
                    snippet = log[-1200:] if len(log) > 1200 else log
                print("\n    ---- error log tail ----")
                for ln in snippet.splitlines()[-20:]:
                    print(f"    {ln}")
                print("    ---- end log tail ----", end="")

            print()


@app.local_entrypoint()
def main():
    print("Loading local config and source files...")
    config = load_config()
    source_files = collect_solution_files(config["build"]["language"])
    print(
        f"Loaded {len(source_files)} source file(s) for "
        f"{config['solution']['name']} ({config['solution']['definition']})."
    )

    print("\nRunning single-workload validation on Modal B200...")
    results = run_benchmark.remote(config, source_files)

    if not results:
        print("No results returned!")
        return

    print_results(results)
