import json
import subprocess
import tempfile
from pathlib import Path

import modal

app = modal.App("torch-profiler-sanity")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.8.0")
)


@app.function(image=image, gpu="B200:1", timeout=1200)
def run_profile() -> dict:
    script = r"""
import json
import torch

assert torch.cuda.is_available(), "CUDA not available"

x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
w = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("/tmp/torch_profile"),
    record_shapes=True,
) as prof:
    for _ in range(4):
        y = x @ w
        torch.cuda.synchronize()
        prof.step()

table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
print(json.dumps({
    "cuda_available": torch.cuda.is_available(),
    "torch_version": torch.__version__,
    "table": table,
}))
"""
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "profile.py"
        target.write_text(script)
        proc = subprocess.run(["python", str(target)], capture_output=True, text=True)
        return {
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-12000:],
            "stderr_tail": proc.stderr[-12000:],
        }


@app.local_entrypoint()
def main():
    out = run_profile.remote()
    print(json.dumps(out, indent=2))
