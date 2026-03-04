"""
FlashInfer-Bench Modal Torch Profiler Runner.

Profiles one selected workload of the submitted solution using torch.profiler on
Modal B200 GPU, then saves a local report.
"""

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import modal

try:
    import tomllib
except ImportError:
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = modal.App("flashinfer-bench-torch-profiler")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"
TRACE_SET_DATA_PATH = "/data/mlsys26-contest"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("flashinfer-bench", "torch==2.8.0", "triton", "numpy")
    .run_commands(
        "python -c \"import pathlib, flashinfer_bench; "
        "p=pathlib.Path(flashinfer_bench.__file__).parent/'bench'/'utils.py'; "
        "s=p.read_text(); old='t = t.contiguous().pin_memory()'; "
        "p.write_text(s.replace(old, 't = t.contiguous()')); "
        "print('patched', p)\""
    )
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


def _normalize_json(value):
    if hasattr(value, "model_dump"):
        return _normalize_json(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _normalize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_json(v) for v in value]
    return value


def _resolve_builder_registry():
    """Resolve builder registry across flashinfer-bench version variants."""
    # Prefer public import path if available.
    try:
        from flashinfer_bench.compile import get_builder_registry

        return get_builder_registry()
    except Exception:
        pass

    # Fall back to module-level registry helpers with varying names.
    import flashinfer_bench.compile as compile_mod
    import flashinfer_bench.compile.registry as reg_mod

    for mod in (reg_mod, compile_mod):
        reg_obj = getattr(mod, "registry", None)
        if reg_obj is not None and callable(getattr(reg_obj, "build", None)):
            return reg_obj

        for fn_name in ("get_builder_registry", "get_registry", "get_builder"):
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn()

    # Newer flashinfer-bench may only expose classes and priority list.
    builder_registry_cls = getattr(reg_mod, "BuilderRegistry", None)
    builder_priority = getattr(reg_mod, "_BUILDER_PRIORITY", None)
    if builder_registry_cls is not None and builder_priority:
        builders = [builder_cls() for builder_cls in builder_priority]
        reg_obj = builder_registry_cls(builders)
        if callable(getattr(reg_obj, "build", None)):
            return reg_obj

    available = sorted(
        set(
            name
            for name in list(dir(compile_mod)) + list(dir(reg_mod))
            if name.startswith("get_") or "registry" in name.lower()
        )
    )
    raise RuntimeError(f"Unable to resolve builder registry. Available hints: {available}")


def _build_runnable_from_registry(registry, definition, solution):
    """Build runnable while handling possible signature variants."""
    build = getattr(registry, "build", None)
    if not callable(build):
        raise RuntimeError("Registry does not expose a callable build method")

    try:
        return build(definition, solution)
    except TypeError:
        return build(solution, definition)


def _close_registry(registry):
    for fn_name in ("clear", "clear_cache"):
        fn = getattr(registry, fn_name, None)
        if callable(fn):
            fn()
            return


def _safe_number(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _get_event_metric(event, names) -> float:
    for name in names:
        if not hasattr(event, name):
            continue
        value = getattr(event, name)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        num = _safe_number(value)
        if num != 0.0:
            return num
    return 0.0


def _invoke_runnable(runnable, inp):
    """Invoke runnable across possible input representations."""
    # kwargs-style inputs
    if isinstance(inp, dict):
        try:
            return runnable(**inp)
        except TypeError:
            return runnable(inp)

    # positional-style inputs
    if isinstance(inp, (list, tuple)):
        last_err = None
        for caller in (lambda: runnable(*inp), lambda: runnable(inp)):
            try:
                return caller()
            except TypeError as e:
                last_err = e
        if last_err is not None:
            raise last_err

    # scalar / unknown object
    return runnable(inp)


def _canonicalize_profile_input(inputs):
    """Unwrap common container wrappers from flashinfer-bench input payloads."""
    cur = inputs
    # Unwrap single-item wrappers like [payload] / (payload,)
    while isinstance(cur, (list, tuple)) and len(cur) == 1:
        first = cur[0]
        if isinstance(first, (dict, list, tuple)):
            cur = first
            continue
        break
    return cur


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_torch_profile(
    config_data: dict,
    source_files: dict,
    workload_index: int = 0,
    warmup_steps: int = 2,
    active_steps: int = 6,
    top_k: int = 20,
) -> dict:
    import os
    import torch
    from flashinfer_bench import BuildSpec, TraceSet
    from flashinfer_bench.agents import pack_solution_from_files
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    from flashinfer_bench.utils import dtype_str_to_torch_dtype

    solution_config = config_data["solution"]
    build_config = config_data["build"]

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
    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    idx = max(0, min(int(workload_index), len(workloads) - 1))
    trace_entry = workloads[idx]
    workload = getattr(trace_entry, "workload", trace_entry)

    os.environ["FIB_DATASET_PATH"] = TRACE_SET_DATA_PATH
    os.chdir(TRACE_SET_DATA_PATH)

    if any(inp.type == "safetensors" for inp in workload.inputs.values()):
        try:
            loaded_stensors = load_safetensors(definition, workload, traceset_root=trace_set.root)
        except TypeError:
            try:
                loaded_stensors = load_safetensors(definition, workload, trace_set.root)
            except TypeError:
                loaded_stensors = load_safetensors(definition, workload)
    else:
        loaded_stensors = {}

    try:
        inputs = gen_inputs(definition, workload, device="cuda:0", stensors=loaded_stensors)
    except TypeError:
        try:
            inputs = gen_inputs(definition, workload, "cuda:0", loaded_stensors)
        except TypeError:
            inputs = gen_inputs(definition, workload, device="cuda:0")

    # Some versions return a single input payload; others return trial input list.
    profile_input = _canonicalize_profile_input(inputs)
    # Older flashinfer-bench gen_inputs may return positional inputs only (without DPS outputs).
    if isinstance(profile_input, (list, tuple)) and len(profile_input) == len(definition.inputs):
        output_shapes = definition.get_output_shapes(workload.axes)
        dps_outputs = []
        for out_idx, (out_name, out_spec) in enumerate(definition.outputs.items()):
            out_dtype = dtype_str_to_torch_dtype(out_spec.dtype)
            if isinstance(output_shapes, dict):
                out_shape = output_shapes.get(out_name)
            elif isinstance(output_shapes, (list, tuple)):
                out_shape = output_shapes[out_idx] if out_idx < len(output_shapes) else None
            else:
                out_shape = None
            if out_shape is None:
                dps_outputs.append(torch.tensor(0, dtype=out_dtype, device="cuda:0"))
            else:
                dps_outputs.append(torch.empty(out_shape, dtype=out_dtype, device="cuda:0"))
        profile_input = list(profile_input) + dps_outputs

    registry = _resolve_builder_registry()
    runnable = _build_runnable_from_registry(registry, definition, solution)

    trace_path = Path("/tmp") / f"torch_profile_{workload.uuid.replace('-', '')}.json"
    try:
        with torch.no_grad():
            for _ in range(max(1, int(warmup_steps))):
                _invoke_runnable(runnable, profile_input)
        torch.cuda.synchronize(device="cuda:0")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for _ in range(max(1, int(active_steps))):
                with torch.no_grad():
                    _invoke_runnable(runnable, profile_input)
                torch.cuda.synchronize(device="cuda:0")

        prof.export_chrome_trace(str(trace_path))

        events = list(prof.key_averages())
        def _self_cuda_us(ev):
            return _get_event_metric(
                ev,
                [
                    "self_cuda_time_total",
                    "self_device_time_total",
                    "self_privateuse1_time_total",
                ],
            )

        def _cuda_total_us(ev):
            return _get_event_metric(
                ev,
                [
                    "cuda_time_total",
                    "device_time_total",
                    "privateuse1_time_total",
                ],
            )

        events.sort(key=_self_cuda_us, reverse=True)
        top_events = []
        for ev in events[: max(1, int(top_k))]:
            self_cuda_us = _self_cuda_us(ev)
            cuda_total_us = _cuda_total_us(ev)
            top_events.append(
                {
                    "name": ev.key,
                    "calls": int(getattr(ev, "count", 0)),
                    "self_cuda_us": float(self_cuda_us),
                    "cuda_total_us": float(cuda_total_us),
                    "self_cpu_us": float(getattr(ev, "self_cpu_time_total", 0.0)),
                    "cpu_total_us": float(getattr(ev, "cpu_time_total", 0.0)),
                }
            )

        total_self_cuda_us = sum(item["self_cuda_us"] for item in top_events) or 1.0
        for item in top_events:
            item["self_cuda_pct_topk"] = (item["self_cuda_us"] / total_self_cuda_us) * 100.0

        trace_size = trace_path.stat().st_size if trace_path.exists() else 0
        try:
            table = prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=max(10, top_k)
            )
        except Exception:
            table = prof.key_averages().table(
                sort_by="self_device_time_total", row_limit=max(10, top_k)
            )

        result = {
            "solution_name": solution.name,
            "definition": solution.definition,
            "workload_index": idx,
            "num_workloads": len(workloads),
            "workload_uuid": str(workload.uuid),
            "workload_axes": _normalize_json(workload.axes),
            "warmup_steps": int(warmup_steps),
            "active_steps": int(active_steps),
            "trace_path": str(trace_path),
            "trace_size_bytes": int(trace_size),
            "top_ops": top_events,
            "profile_table": table,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        return json.loads(json.dumps(result, default=str))
    finally:
        if hasattr(runnable, "close") and callable(runnable.close):
            runnable.close()
        _close_registry(registry)


def _write_report_files(result: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"torch_profile_wl{result['workload_index']:02d}_{result['workload_uuid'][:8]}_{ts}"

    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"

    json_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))

    lines = []
    lines.append(f"# Torch Profiler Report: {result['solution_name']}")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- Definition: `{result['definition']}`")
    lines.append(f"- Workload UUID: `{result['workload_uuid']}`")
    lines.append(f"- Workload index: `{result['workload_index']}` / `{result['num_workloads'] - 1}`")
    lines.append(f"- Axes: `{result['workload_axes']}`")
    lines.append(f"- Torch: `{result['torch_version']}`")
    lines.append(f"- CUDA device: `{result['cuda_device_name']}`")
    lines.append(f"- Warmup steps: `{result['warmup_steps']}`")
    lines.append(f"- Active steps: `{result['active_steps']}`")
    lines.append(f"- Chrome trace size: `{result['trace_size_bytes']}` bytes")
    lines.append("")
    lines.append("## Top CUDA Ops (by self_cuda_us)")
    lines.append("")
    lines.append("| # | Name | Calls | Self CUDA (us) | CUDA Total (us) | Share (top-k) |")
    lines.append("|---|------|------:|---------------:|----------------:|--------------:|")
    for i, op in enumerate(result.get("top_ops", []), 1):
        lines.append(
            f"| {i} | `{op['name']}` | {op['calls']} | {op['self_cuda_us']:.3f} | "
            f"{op['cuda_total_us']:.3f} | {op['self_cuda_pct_topk']:.2f}% |"
        )
    lines.append("")
    lines.append("## Full Profiler Table")
    lines.append("")
    lines.append("```")
    lines.append(result.get("profile_table", ""))
    lines.append("```")
    lines.append("")
    lines.append("## Raw JSON")
    lines.append(f"- `{json_path.name}`")

    md_path.write_text("\n".join(lines))
    return json_path, md_path


@app.local_entrypoint()
def main(
    workload_index: int = 0,
    warmup_steps: int = 2,
    active_steps: int = 6,
    top_k: int = 20,
    out_dir: str = "results/profiles",
):
    print("Loading local config and source files...")
    config = load_config()
    source_files = collect_solution_files(config["build"]["language"])
    print(
        f"Loaded {len(source_files)} source file(s) for "
        f"{config['solution']['name']} ({config['solution']['definition']})."
    )

    print("\nRunning torch.profiler on Modal B200...")
    result = run_torch_profile.remote(
        config,
        source_files,
        workload_index=workload_index,
        warmup_steps=warmup_steps,
        active_steps=active_steps,
        top_k=top_k,
    )

    target_dir = PROJECT_ROOT / out_dir
    json_path, md_path = _write_report_files(result, target_dir)

    print("\nTorch profiler result summary:")
    print(
        json.dumps(
            {
                "definition": result["definition"],
                "workload_uuid": result["workload_uuid"],
                "workload_index": result["workload_index"],
                "torch_version": result["torch_version"],
                "cuda_device_name": result["cuda_device_name"],
                "top_ops_count": len(result.get("top_ops", [])),
                "json_report": str(json_path),
                "md_report": str(md_path),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
