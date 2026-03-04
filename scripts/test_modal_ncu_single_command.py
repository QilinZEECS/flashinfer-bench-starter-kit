import json
import subprocess
import tempfile
from pathlib import Path

import modal

app = modal.App("ncu-single-command")

image = modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")


def _run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "cmd": " ".join(cmd),
        "returncode": p.returncode,
        "stdout_tail": p.stdout[-6000:],
        "stderr_tail": p.stderr[-6000:],
    }


@app.function(image=image, gpu="B200:1", timeout=1200)
def run_one(mode: str = "detailed_all") -> dict:
    cu_src = r"""
#include <cstdio>
#include <cuda_runtime.h>

__global__ void saxpy(int n, float a, const float* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
  const int n = 1 << 20;
  const size_t bytes = n * sizeof(float);
  float *x, *y;
  float *hx = (float*)malloc(bytes);
  float *hy = (float*)malloc(bytes);
  for (int i = 0; i < n; ++i) { hx[i] = 1.0f; hy[i] = 2.0f; }
  cudaMalloc(&x, bytes);
  cudaMalloc(&y, bytes);
  cudaMemcpy(x, hx, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(y, hy, bytes, cudaMemcpyHostToDevice);
  saxpy<<<(n + 255) / 256, 256>>>(n, 2.0f, x, y);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("cuda error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  cudaMemcpy(hy, y, bytes, cudaMemcpyDeviceToHost);
  printf("y0=%f\n", hy[0]);
  cudaFree(x);
  cudaFree(y);
  free(hx);
  free(hy);
  return 0;
}
"""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        cu = td_path / "saxpy.cu"
        binp = td_path / "saxpy"
        cu.write_text(cu_src)

        build = _run(["nvcc", "-O3", str(cu), "-o", str(binp)])
        if build["returncode"] != 0:
            return {"phase": "build_failed", "build": build}

        mode_to_cmd = {
            "list_sets": ["ncu", "--list-sets"],
            "version": ["ncu", "--version"],
            "default_all": ["ncu", "--target-processes", "all", str(binp)],
            "detailed_all": ["ncu", "--set", "detailed", "--target-processes", "all", str(binp)],
            "full_all": ["ncu", "--set", "full", "--target-processes", "all", str(binp)],
            "launchstats_all": ["ncu", "--section", "LaunchStats", "--target-processes", "all", str(binp)],
            "speedoflight_all": ["ncu", "--set", "speedOfLight", "--target-processes", "all", str(binp)],
            "detailed_app": ["ncu", "--set", "detailed", "--target-processes", "application-only", str(binp)],
            "detailed_cache_none": [
                "ncu",
                "--set",
                "detailed",
                "--cache-control",
                "none",
                "--target-processes",
                "all",
                str(binp),
            ],
        }
        if mode not in mode_to_cmd:
            return {"phase": "invalid_mode", "mode": mode, "valid_modes": sorted(mode_to_cmd)}

        plain = _run([str(binp)])
        chosen = _run(mode_to_cmd[mode])
        return {"phase": "done", "mode": mode, "plain_run": plain, "ncu_run": chosen}


@app.local_entrypoint()
def main(mode: str = "detailed_all"):
    out = run_one.remote(mode)
    print(json.dumps(out, indent=2))
