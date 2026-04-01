"""
FlashInfer-Bench Modal NCU Profiler.

Runs Nsight Compute profiling on Modal B200 GPUs. The local machine only needs
`modal`; all flashinfer-bench dependencies run remotely in the Modal container.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import modal

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = modal.App("flashinfer-bench-ncu")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"
TRACE_SET_DATA_PATH = "/data/mlsys26-contest"
CUTLASS_INCLUDE = "/opt/cutlass/include"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .run_commands(
        "apt-get update && apt-get install -y git && "
        "git clone --depth 1 https://github.com/NVIDIA/cutlass.git /opt/cutlass"
    )
    .pip_install("flashinfer-bench", "torch==2.8.0", "triton", "numpy")
    .run_commands(
        "python -c \"import pathlib, flashinfer_bench; "
        "p=pathlib.Path(flashinfer_bench.__file__).parent/'bench'/'utils.py'; "
        "s=p.read_text(); old='t = t.contiguous().pin_memory()'; "
        "assert old in s, f'missing pattern in {p}'; "
        "p.write_text(s.replace(old, 't = t.contiguous()')); "
        "print('patched', p)\""
    )
    .run_commands(
        # Remove NVTX filter from NCU command so all kernels are profiled
        'python -c "'
        "import pathlib, flashinfer_bench;"
        "p=pathlib.Path(flashinfer_bench.__file__).parent/'agents'/'ncu.py';"
        "t=p.read_text();"
        "t=t.replace('        \\x22--nvtx\\x22,\\n        \\x22--nvtx-include\\x22,\\n        \\x22flashinfer_bench_ncu_profile\\x22,\\n','');"
        "p.write_text(t);"
        "assert '--nvtx' not in p.read_text(), 'PATCH FAILED: --nvtx still present';"
        "import subprocess; subprocess.run(['find', str(p.parent), '-name', '*.pyc', '-delete']);"
        "print('patched ncu.py v3, verified --nvtx removed');"
        '"'
    )
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


def _normalize_json(value):
    if hasattr(value, "model_dump"):
        return _normalize_json(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _normalize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_json(v) for v in value]
    return value


def _workload_identifier(workload) -> str:
    if hasattr(workload, "uuid"):
        return str(workload.uuid)
    if hasattr(workload, "id"):
        return str(workload.id)
    if hasattr(workload, "name"):
        return str(workload.name)
    if hasattr(workload, "model_dump"):
        dumped = workload.model_dump()
        for key in ("uuid", "id", "name"):
            if key in dumped:
                return str(dumped[key])
        return json.dumps(dumped, default=str)[:256]
    return str(workload)[:256]


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_ncu_profile(
    config_data: dict,
    source_files: dict,
    set_name: str = "detailed",
    page: str = "details",
    workload_index: int = 0,
) -> dict:
    """Pack solution and run NCU profiling on one workload."""
    os.environ["CPATH"] = (
        CUTLASS_INCLUDE
        if not os.environ.get("CPATH")
        else f"{CUTLASS_INCLUDE}:{os.environ['CPATH']}"
    )
    from flashinfer_bench import BuildSpec, TraceSet
    from flashinfer_bench.agents import flashinfer_bench_run_ncu, pack_solution_from_files

    solution_config = config_data["solution"]
    build_config = config_data["build"]

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
            name=solution_config["name"],
            definition=solution_config["definition"],
            author=solution_config["author"],
        )

    trace_set = TraceSet.from_path(TRACE_SET_DATA_PATH)
    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    idx = max(0, min(int(workload_index), len(workloads) - 1))
    trace_entry = workloads[idx]
    workload = getattr(trace_entry, "workload", trace_entry)

    # flashinfer_bench_run_ncu looks up definitions via FIB_DATASET_PATH.
    os.environ["FIB_DATASET_PATH"] = TRACE_SET_DATA_PATH
    os.chdir(TRACE_SET_DATA_PATH)

    output = flashinfer_bench_run_ncu(
        solution=solution,
        workload=workload,
        set=set_name,
        page=page,
        timeout=900,
    )

    return {
        "solution_name": solution.name,
        "definition": solution.definition,
        "workload_index": idx,
        "num_workloads": len(workloads),
        "workload": _workload_identifier(workload),
        "ncu_output": _normalize_json(output),
    }


@app.local_entrypoint()
def main(set_name: str = "detailed", page: str = "details", workload_index: int = 0):
    """Run NCU profile and print results."""
    print("Loading local config and source files...")
    config = load_config()
    source_files = collect_solution_files(config["build"]["language"])
    print(
        f"Loaded {len(source_files)} source file(s) for "
        f"{config['solution']['name']} ({config['solution']['definition']})."
    )

    print("\nRunning NCU profiling on Modal B200...")
    result = run_ncu_profile.remote(config, source_files, set_name, page, workload_index)
    print("\nNCU result:")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
