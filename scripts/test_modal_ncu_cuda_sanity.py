import json
import subprocess
import tempfile
from pathlib import Path

import modal

app = modal.App("ncu-cuda-sanity")

image = modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")


@app.function(image=image, gpu="B200:1", timeout=1800)
def run_sanity() -> dict:
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

        build = subprocess.run(
            ["nvcc", "-O3", str(cu), "-o", str(binp)],
            capture_output=True,
            text=True,
        )
        if build.returncode != 0:
            return {
                "phase": "build",
                "returncode": build.returncode,
                "stdout": build.stdout[-8000:],
                "stderr": build.stderr[-8000:],
            }

        run_plain = subprocess.run([str(binp)], capture_output=True, text=True)
        run_ncu_app = subprocess.run(
            ["ncu", "--set", "detailed", "--target-processes", "application-only", str(binp)],
            capture_output=True,
            text=True,
        )
        run_ncu_all = subprocess.run(
            ["ncu", "--set", "detailed", "--target-processes", "all", str(binp)],
            capture_output=True,
            text=True,
        )
        return {
            "phase": "done",
            "plain_returncode": run_plain.returncode,
            "plain_stdout": run_plain.stdout[-8000:],
            "plain_stderr": run_plain.stderr[-8000:],
            "ncu_app_returncode": run_ncu_app.returncode,
            "ncu_app_stdout": run_ncu_app.stdout[-8000:],
            "ncu_app_stderr": run_ncu_app.stderr[-8000:],
            "ncu_all_returncode": run_ncu_all.returncode,
            "ncu_all_stdout": run_ncu_all.stdout[-8000:],
            "ncu_all_stderr": run_ncu_all.stderr[-8000:],
        }


@app.local_entrypoint()
def main():
    out = run_sanity.remote()
    print(json.dumps(out, indent=2))
