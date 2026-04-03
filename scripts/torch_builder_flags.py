"""Helpers for CUDA build flags — passthrough for Triton solutions."""


def build_name_with_cuda_flags(name: str, build_config: dict) -> str:
    cuda_arch = build_config.get("cuda_arch")
    if cuda_arch:
        return f"{name}_{cuda_arch}"
    return name


def maybe_patch_torch_cpp_extension(build_config: dict) -> list[str]:
    cuda_cflags = build_config.get("extra_cuda_cflags")
    if not cuda_cflags:
        return []
    import torch.utils.cpp_extension as cpp_ext

    orig_build = cpp_ext.BuildExtension

    class PatchedBuildExtension(orig_build):
        def build_extensions(self):
            for ext in self.extensions:
                if hasattr(ext, "extra_compile_args") and "nvcc" in ext.extra_compile_args:
                    ext.extra_compile_args["nvcc"].extend(cuda_cflags)
            super().build_extensions()

    cpp_ext.BuildExtension = PatchedBuildExtension
    return cuda_cflags
