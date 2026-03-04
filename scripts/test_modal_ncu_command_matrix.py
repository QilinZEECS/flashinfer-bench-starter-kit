import json
import subprocess
import tempfile
from pathlib import Path

import modal

app = modal.App("ncu-command-matrix")

image = modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")


def _run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


@app.function(image=image, gpu="B200:1", timeout=1800)
def run_matrix() -> dict:
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

        checks = {
            "ncu_version": _run(["ncu", "--version"]),
            "ncu_list_sets": _run(["ncu", "--list-sets"]),
            "plain_run": _run([str(binp)]),
            "ncu_default_all": _run(["ncu", "--target-processes", "all", str(binp)]),
            "ncu_detailed_all": _run(["ncu", "--set", "detailed", "--target-processes", "all", str(binp)]),
            "ncu_full_all": _run(["ncu", "--set", "full", "--target-processes", "all", str(binp)]),
            "ncu_launchstats_all": _run(
                ["ncu", "--section", "LaunchStats", "--target-processes", "all", str(binp)]
            ),
            "ncu_speedoflight_all": _run(
                ["ncu", "--set", "speedOfLight", "--target-processes", "all", str(binp)]
            ),
            "ncu_detailed_app": _run(
                ["ncu", "--set", "detailed", "--target-processes", "application-only", str(binp)]
            ),
            "ncu_no_cache_control": _run(
                ["ncu", "--set", "detailed", "--cache-control", "none", "--target-processes", "all", str(binp)]
            ),
        }
        return {"phase": "done", "checks": checks}


@app.local_entrypoint()
def main():
    out = run_matrix.remote()
    print(json.dumps(out, indent=2))
