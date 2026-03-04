import json
import subprocess
import tempfile
from pathlib import Path

import modal

app = modal.App("ncu-sanity")

image = modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12").pip_install(
    "torch==2.8.0"
)


@app.function(image=image, gpu="B200:1", timeout=1800)
def run_sanity() -> dict:
    script = """
import torch
a = torch.randn(2048, 2048, device='cuda', dtype=torch.float16)
b = torch.randn(2048, 2048, device='cuda', dtype=torch.float16)
c = a @ b
torch.cuda.synchronize()
print(float(c[0,0].item()))
"""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        target = td_path / "sanity.py"
        target.write_text(script)
        cmd = ["ncu", "--set", "detailed", "--target-processes", "all", "python", str(target)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
        }


@app.local_entrypoint()
def main():
    out = run_sanity.remote()
    print(json.dumps(out, indent=2))
