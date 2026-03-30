#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#if defined(__has_include)
#if __has_include(<cutlass/cutlass.h>)
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/layout/matrix.h>
#define FLASHINFER_HAS_CUTLASS 1
#else
#define FLASHINFER_HAS_CUTLASS 0
#endif
#else
#define FLASHINFER_HAS_CUTLASS 0
#endif

#if FLASHINFER_HAS_CUTLASS && defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#define FLASHINFER_HAS_CUTLASS_SM100 1
#else
#define FLASHINFER_HAS_CUTLASS_SM100 0
#endif

namespace py = pybind11;

namespace {

constexpr int64_t BLOCK = 128;
constexpr int64_t TOP_K = 8;
constexpr int64_t N_GROUP = 8;
constexpr int64_t TOPK_GROUP = 4;
constexpr int64_t E_GLOBAL = 256;

// ── Scalar helpers ───────────────────────────────────────────────────────────

int64_t to_int64(const py::object& obj) {
  if (py::isinstance<torch::Tensor>(obj)) {
    return obj.cast<torch::Tensor>().item<int64_t>();
  }
  return obj.cast<int64_t>();
}

double to_double(const py::object& obj) {
  if (py::isinstance<torch::Tensor>(obj)) {
    return obj.cast<torch::Tensor>().item<double>();
  }
  return obj.cast<double>();
}

struct GroupedGemmProblem {
  int64_t group_idx;
  int64_t m;
  int64_t n;
  int64_t k;
};

// ── CUTLASS SM100 FP8 Grouped GEMM ─────────────────────────────────────────

#if FLASHINFER_HAS_CUTLASS_SM100

template <typename T>
bool cuda_alloc_and_copy(T*& device_ptr, const std::vector<T>& host_vec) {
  if (host_vec.empty()) {
    device_ptr = nullptr;
    return true;
  }
  if (cudaMalloc(reinterpret_cast<void**>(&device_ptr),
                 sizeof(T) * host_vec.size()) != cudaSuccess) {
    return false;
  }
  return cudaMemcpy(device_ptr, host_vec.data(),
                    sizeof(T) * host_vec.size(),
                    cudaMemcpyHostToDevice) == cudaSuccess;
}

bool is_fp8_e4m3_tensor(const torch::Tensor& tensor) {
  return tensor.scalar_type() == torch::kFloat8_e4m3fn;
}

// GPU kernel for scale packing — eliminates CPU↔GPU transfers
template <class Layout>
__global__ void pack_scales_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    Layout layout,
    int rows,
    int k_blocks,
    int row_gran) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * k_blocks) return;
  int row = idx / k_blocks;
  int k_blk = idx % k_blocks;
  dst[layout(row * row_gran, k_blk * 128, 0)] = src[row * k_blocks + k_blk];
}

template <class Layout>
torch::Tensor pack_block_scale_tensor(
    const torch::Tensor& logical_scale,
    Layout layout,
    int64_t logical_rows,
    int64_t logical_k_blocks,
    int64_t row_granularity) {
  auto src = logical_scale.to(torch::kFloat32).contiguous();
  auto packed = torch::zeros(
      {static_cast<int64_t>(cute::cosize(layout))},
      torch::TensorOptions().dtype(torch::kFloat32).device(src.device()));

  int total = static_cast<int>(logical_rows * logical_k_blocks);
  if (total > 0 && src.is_cuda()) {
    constexpr int threads = 256;
    int blocks = (total + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();
    pack_scales_kernel<<<blocks, threads, 0, stream.stream()>>>(
        src.data_ptr<float>(),
        packed.data_ptr<float>(),
        layout,
        static_cast<int>(logical_rows),
        static_cast<int>(logical_k_blocks),
        static_cast<int>(row_granularity));
  } else {
    // CPU fallback
    auto src_ptr = src.data_ptr<float>();
    auto packed_tensor = cute::make_tensor(packed.data_ptr<float>(), layout);
    for (int64_t row = 0; row < logical_rows; ++row) {
      for (int64_t k_blk = 0; k_blk < logical_k_blocks; ++k_blk) {
        packed_tensor(row * row_granularity, k_blk * BLOCK, 0) =
            src_ptr[row * logical_k_blocks + k_blk];
      }
    }
  }

  return packed;
}

// ── GEMM1 SM100 FP8 type aliases ────────────────────────────────────────────

using NativeGemm1ProblemShape =
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
using NativeGemm1ElementA = cutlass::float_e4m3_t;
using NativeGemm1LayoutA = cutlass::layout::RowMajor;
constexpr int NativeGemm1AlignmentA =
    128 / cutlass::sizeof_bits<NativeGemm1ElementA>::value;

using NativeGemm1ElementB = cutlass::float_e4m3_t;
using NativeGemm1LayoutB = cutlass::layout::ColumnMajor;
constexpr int NativeGemm1AlignmentB =
    128 / cutlass::sizeof_bits<NativeGemm1ElementB>::value;

using NativeGemm1ElementC = float;
using NativeGemm1LayoutC = cutlass::layout::RowMajor;
constexpr int NativeGemm1AlignmentC =
    128 / cutlass::sizeof_bits<NativeGemm1ElementC>::value;

using NativeGemm1ElementD = NativeGemm1ElementC;
using NativeGemm1LayoutD = NativeGemm1LayoutC;
constexpr int NativeGemm1AlignmentD = NativeGemm1AlignmentC;

using NativeGemm1ElementAccumulator = float;
using NativeGemm1ElementCompute = float;
using NativeGemm1MmaTileShape =
    cute::Shape<cute::_128, cute::_128, cute::_128>;
using NativeGemm1ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
using NativeGemm1ScaleConfig =
    cutlass::detail::Sm100BlockwiseScaleConfig<1, 128, 128>;
using NativeGemm1LayoutSFA =
    decltype(NativeGemm1ScaleConfig::deduce_layoutSFA());
using NativeGemm1LayoutSFB =
    decltype(NativeGemm1ScaleConfig::deduce_layoutSFB());

using NativeGemm1CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        NativeGemm1MmaTileShape,
        NativeGemm1ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        NativeGemm1ElementAccumulator,
        NativeGemm1ElementCompute,
        NativeGemm1ElementC,
        NativeGemm1LayoutC*,
        NativeGemm1AlignmentC,
        NativeGemm1ElementD,
        NativeGemm1LayoutD*,
        NativeGemm1AlignmentD,
        cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>::CollectiveOp;

using NativeGemm1CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        NativeGemm1ElementA,
        cute::tuple<NativeGemm1LayoutA*, NativeGemm1LayoutSFA*>,
        NativeGemm1AlignmentA,
        NativeGemm1ElementB,
        cute::tuple<NativeGemm1LayoutB*, NativeGemm1LayoutSFB*>,
        NativeGemm1AlignmentB,
        NativeGemm1ElementAccumulator,
        NativeGemm1MmaTileShape,
        NativeGemm1ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename NativeGemm1CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::
            KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100>::CollectiveOp;

using NativeGemm1Kernel = cutlass::gemm::kernel::GemmUniversal<
    NativeGemm1ProblemShape,
    NativeGemm1CollectiveMainloop,
    NativeGemm1CollectiveEpilogue,
    void>;
using NativeGemm1 =
    cutlass::gemm::device::GemmUniversalAdapter<NativeGemm1Kernel>;
using NativeGemm1UnderlyingProblemShape =
    typename NativeGemm1ProblemShape::UnderlyingProblemShape;
using NativeGemm1StrideA = typename NativeGemm1::GemmKernel::InternalStrideA;
using NativeGemm1StrideB = typename NativeGemm1::GemmKernel::InternalStrideB;
using NativeGemm1StrideC = typename NativeGemm1::GemmKernel::InternalStrideC;
using NativeGemm1StrideD = typename NativeGemm1::GemmKernel::InternalStrideD;

// ── GEMM2 SM100 FP8 type aliases ────────────────────────────────────────────

using NativeGemm2ProblemShape =
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
using NativeGemm2ElementA = cutlass::float_e4m3_t;
using NativeGemm2LayoutA = cutlass::layout::RowMajor;
constexpr int NativeGemm2AlignmentA =
    128 / cutlass::sizeof_bits<NativeGemm2ElementA>::value;

using NativeGemm2ElementB = cutlass::float_e4m3_t;
using NativeGemm2LayoutB = cutlass::layout::ColumnMajor;
constexpr int NativeGemm2AlignmentB =
    128 / cutlass::sizeof_bits<NativeGemm2ElementB>::value;

using NativeGemm2ElementC = float;
using NativeGemm2LayoutC = cutlass::layout::RowMajor;
constexpr int NativeGemm2AlignmentC =
    128 / cutlass::sizeof_bits<NativeGemm2ElementC>::value;

using NativeGemm2ElementD = NativeGemm2ElementC;
using NativeGemm2LayoutD = NativeGemm2LayoutC;
constexpr int NativeGemm2AlignmentD = NativeGemm2AlignmentC;

using NativeGemm2ElementAccumulator = float;
using NativeGemm2ElementCompute = float;
using NativeGemm2MmaTileShape =
    cute::Shape<cute::_128, cute::_128, cute::_128>;
using NativeGemm2ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
using NativeGemm2ScaleConfig =
    cutlass::detail::Sm100BlockwiseScaleConfig<1, 128, 128>;
using NativeGemm2LayoutSFA =
    decltype(NativeGemm2ScaleConfig::deduce_layoutSFA());
using NativeGemm2LayoutSFB =
    decltype(NativeGemm2ScaleConfig::deduce_layoutSFB());

using NativeGemm2CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        NativeGemm2MmaTileShape,
        NativeGemm2ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        NativeGemm2ElementAccumulator,
        NativeGemm2ElementCompute,
        NativeGemm2ElementC,
        NativeGemm2LayoutC*,
        NativeGemm2AlignmentC,
        NativeGemm2ElementD,
        NativeGemm2LayoutD*,
        NativeGemm2AlignmentD,
        cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>::CollectiveOp;

using NativeGemm2CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        NativeGemm2ElementA,
        cute::tuple<NativeGemm2LayoutA*, NativeGemm2LayoutSFA*>,
        NativeGemm2AlignmentA,
        NativeGemm2ElementB,
        cute::tuple<NativeGemm2LayoutB*, NativeGemm2LayoutSFB*>,
        NativeGemm2AlignmentB,
        NativeGemm2ElementAccumulator,
        NativeGemm2MmaTileShape,
        NativeGemm2ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename NativeGemm2CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::
            KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100>::CollectiveOp;

using NativeGemm2Kernel = cutlass::gemm::kernel::GemmUniversal<
    NativeGemm2ProblemShape,
    NativeGemm2CollectiveMainloop,
    NativeGemm2CollectiveEpilogue,
    void>;
using NativeGemm2 =
    cutlass::gemm::device::GemmUniversalAdapter<NativeGemm2Kernel>;
using NativeGemm2UnderlyingProblemShape =
    typename NativeGemm2ProblemShape::UnderlyingProblemShape;
using NativeGemm2StrideA = typename NativeGemm2::GemmKernel::InternalStrideA;
using NativeGemm2StrideB = typename NativeGemm2::GemmKernel::InternalStrideB;
using NativeGemm2StrideC = typename NativeGemm2::GemmKernel::InternalStrideC;
using NativeGemm2StrideD = typename NativeGemm2::GemmKernel::InternalStrideD;

// ── GEMM1 validation & launch ───────────────────────────────────────────────

enum class Gemm1NativeSupportCode {
  kSupported,
  kTensorProperties,
  kShapeAlignment,
  kMinM,
  kScaleLayout,
};

bool gemm1_native_cutlass_inputs_available(
    const torch::Tensor& a_fp8_padded,
    const torch::Tensor& a_scale_padded,
    const torch::Tensor& w1_fp8_active,
    const torch::Tensor& w1_scale_active) {
  if (!a_fp8_padded.is_cuda() || !a_scale_padded.is_cuda() ||
      !w1_fp8_active.is_cuda() || !w1_scale_active.is_cuda()) {
    return false;
  }
  if (!a_fp8_padded.is_contiguous() || !a_scale_padded.is_contiguous() ||
      !w1_fp8_active.is_contiguous() || !w1_scale_active.is_contiguous()) {
    return false;
  }
  if (a_scale_padded.scalar_type() != torch::kFloat32 ||
      w1_scale_active.scalar_type() != torch::kFloat32) {
    return false;
  }
  if (!is_fp8_e4m3_tensor(a_fp8_padded) || !is_fp8_e4m3_tensor(w1_fp8_active)) {
    return false;
  }
  if (a_fp8_padded.size(2) % BLOCK != 0 || w1_fp8_active.size(1) % BLOCK != 0 ||
      w1_fp8_active.size(2) % BLOCK != 0) {
    return false;
  }
  return true;
}

Gemm1NativeSupportCode gemm1_native_cutlass_problem_support(
    const GroupedGemmProblem& problem) {
  if (problem.n % BLOCK != 0 || problem.k % BLOCK != 0) {
    return Gemm1NativeSupportCode::kShapeAlignment;
  }
  if (problem.m < 16) {
    return Gemm1NativeSupportCode::kMinM;
  }

  auto layout_sfa = NativeGemm1ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(
          static_cast<int>(problem.m),
          static_cast<int>(problem.n),
          static_cast<int>(problem.k),
          1));
  auto layout_sfb = NativeGemm1ScaleConfig::tile_atom_to_shape_SFB(
      cute::make_shape(
          static_cast<int>(problem.m),
          static_cast<int>(problem.n),
          static_cast<int>(problem.k),
          1));

  int64_t expected_sfa_elems = static_cast<int64_t>(cute::cosize(layout_sfa));
  int64_t expected_sfb_elems = static_cast<int64_t>(cute::cosize(layout_sfb));
  int64_t raw_sfa_elems = problem.m * (problem.k / BLOCK);
  int64_t raw_sfb_elems = (problem.n / BLOCK) * (problem.k / BLOCK);
  if (expected_sfa_elems != raw_sfa_elems ||
      expected_sfb_elems != raw_sfb_elems) {
    return Gemm1NativeSupportCode::kScaleLayout;
  }

  return Gemm1NativeSupportCode::kSupported;
}

torch::Tensor run_gemm1_grouped_cutlass_fp8(
    const torch::Tensor& a_fp8_padded,
    const torch::Tensor& a_scale_padded,
    const torch::Tensor& w1_fp8_active,
    const torch::Tensor& w1_scale_active,
    const std::vector<GroupedGemmProblem>& problems) {
  auto num_groups = static_cast<int>(problems.size());
  if (num_groups == 0) {
    return torch::zeros(
        {0, a_fp8_padded.size(1), w1_fp8_active.size(1)},
        a_scale_padded.options().dtype(torch::kFloat32));
  }
  if (!gemm1_native_cutlass_inputs_available(
          a_fp8_padded, a_scale_padded, w1_fp8_active, w1_scale_active)) {
    return torch::Tensor();
  }
  for (const auto& problem : problems) {
    if (gemm1_native_cutlass_problem_support(problem) !=
        Gemm1NativeSupportCode::kSupported) {
      return torch::Tensor();
    }
  }
  if (problems.size() != static_cast<size_t>(a_fp8_padded.size(0))) {
    return torch::Tensor();
  }

  auto max_count = a_fp8_padded.size(1);
  auto gemm1_out = w1_fp8_active.size(1);
  auto opts_f32 = torch::TensorOptions()
                      .dtype(torch::kFloat32)
                      .device(a_fp8_padded.device());
  auto out = torch::zeros({num_groups, max_count, gemm1_out}, opts_f32);

  std::vector<NativeGemm1UnderlyingProblemShape> problem_sizes_host;
  std::vector<const NativeGemm1ElementA*> ptr_a_host;
  std::vector<const NativeGemm1ElementB*> ptr_b_host;
  std::vector<const NativeGemm1ElementAccumulator*> ptr_sfa_host;
  std::vector<const NativeGemm1ElementAccumulator*> ptr_sfb_host;
  std::vector<const float*> ptr_c_host;
  std::vector<float*> ptr_d_host;
  std::vector<NativeGemm1StrideA> stride_a_host;
  std::vector<NativeGemm1StrideB> stride_b_host;
  std::vector<NativeGemm1StrideC> stride_c_host;
  std::vector<NativeGemm1StrideD> stride_d_host;
  std::vector<NativeGemm1LayoutSFA> layout_sfa_host;
  std::vector<NativeGemm1LayoutSFB> layout_sfb_host;
  std::vector<torch::Tensor> packed_sfa_tensors;
  std::vector<torch::Tensor> packed_sfb_tensors;

  problem_sizes_host.reserve(num_groups);
  ptr_a_host.reserve(num_groups);
  ptr_b_host.reserve(num_groups);
  ptr_sfa_host.reserve(num_groups);
  ptr_sfb_host.reserve(num_groups);
  ptr_c_host.reserve(num_groups);
  ptr_d_host.reserve(num_groups);
  stride_a_host.reserve(num_groups);
  stride_b_host.reserve(num_groups);
  stride_c_host.reserve(num_groups);
  stride_d_host.reserve(num_groups);
  layout_sfa_host.reserve(num_groups);
  layout_sfb_host.reserve(num_groups);
  packed_sfa_tensors.reserve(num_groups);
  packed_sfb_tensors.reserve(num_groups);

  for (const auto& problem : problems) {
    auto m = static_cast<int>(problem.m);
    auto n = static_cast<int>(problem.n);
    auto k = static_cast<int>(problem.k);
    auto logical_k_blocks = static_cast<int64_t>(k) / BLOCK;
    auto logical_n_blocks = static_cast<int64_t>(n) / BLOCK;
    auto layout_sfa = NativeGemm1ScaleConfig::tile_atom_to_shape_SFA(
        cute::make_shape(m, n, k, 1));
    auto layout_sfb = NativeGemm1ScaleConfig::tile_atom_to_shape_SFB(
        cute::make_shape(m, n, k, 1));
    auto packed_sfa = pack_block_scale_tensor(
        a_scale_padded[problem.group_idx].slice(0, 0, m),
        layout_sfa, m, logical_k_blocks, /*row_granularity=*/1);
    auto packed_sfb = pack_block_scale_tensor(
        w1_scale_active[problem.group_idx],
        layout_sfb, logical_n_blocks, logical_k_blocks,
        /*row_granularity=*/BLOCK);

    problem_sizes_host.push_back({m, n, k});
    ptr_a_host.push_back(reinterpret_cast<const NativeGemm1ElementA*>(
        a_fp8_padded[problem.group_idx].data_ptr()));
    ptr_b_host.push_back(reinterpret_cast<const NativeGemm1ElementB*>(
        w1_fp8_active[problem.group_idx].data_ptr()));
    packed_sfa_tensors.push_back(std::move(packed_sfa));
    packed_sfb_tensors.push_back(std::move(packed_sfb));
    ptr_sfa_host.push_back(packed_sfa_tensors.back().data_ptr<float>());
    ptr_sfb_host.push_back(packed_sfb_tensors.back().data_ptr<float>());
    ptr_c_host.push_back(out[problem.group_idx].data_ptr<float>());
    ptr_d_host.push_back(out[problem.group_idx].data_ptr<float>());

    stride_a_host.push_back(NativeGemm1StrideA{
        static_cast<int64_t>(k), cute::Int<1>{}, cute::Int<0>{}});
    stride_b_host.push_back(NativeGemm1StrideB{
        static_cast<int64_t>(k), cute::Int<1>{}, cute::Int<0>{}});
    stride_c_host.push_back(NativeGemm1StrideC{
        static_cast<int64_t>(n), cute::Int<1>{}, cute::Int<0>{}});
    stride_d_host.push_back(NativeGemm1StrideD{
        static_cast<int64_t>(n), cute::Int<1>{}, cute::Int<0>{}});
    layout_sfa_host.push_back(layout_sfa);
    layout_sfb_host.push_back(layout_sfb);
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cutlass::Status status = cutlass::Status::kSuccess;

  NativeGemm1UnderlyingProblemShape* problem_sizes_device = nullptr;
  const NativeGemm1ElementA** ptr_a_device = nullptr;
  const NativeGemm1ElementB** ptr_b_device = nullptr;
  const NativeGemm1ElementAccumulator** ptr_sfa_device = nullptr;
  const NativeGemm1ElementAccumulator** ptr_sfb_device = nullptr;
  const NativeGemm1ElementC** ptr_c_device = nullptr;
  NativeGemm1ElementD** ptr_d_device = nullptr;
  NativeGemm1StrideA* stride_a_device = nullptr;
  NativeGemm1StrideB* stride_b_device = nullptr;
  NativeGemm1StrideC* stride_c_device = nullptr;
  NativeGemm1StrideD* stride_d_device = nullptr;
  NativeGemm1LayoutSFA* layout_sfa_device = nullptr;
  NativeGemm1LayoutSFB* layout_sfb_device = nullptr;
  uint8_t* workspace_device = nullptr;

  auto cleanup = [&]() {
    if (workspace_device) cudaFree(workspace_device);
    if (layout_sfb_device) cudaFree(layout_sfb_device);
    if (layout_sfa_device) cudaFree(layout_sfa_device);
    if (stride_d_device) cudaFree(stride_d_device);
    if (stride_c_device) cudaFree(stride_c_device);
    if (stride_b_device) cudaFree(stride_b_device);
    if (stride_a_device) cudaFree(stride_a_device);
    if (ptr_d_device) cudaFree(ptr_d_device);
    if (ptr_c_device) cudaFree(ptr_c_device);
    if (ptr_sfb_device) cudaFree(ptr_sfb_device);
    if (ptr_sfa_device) cudaFree(ptr_sfa_device);
    if (ptr_b_device) cudaFree(ptr_b_device);
    if (ptr_a_device) cudaFree(ptr_a_device);
    if (problem_sizes_device) cudaFree(problem_sizes_device);
  };

  if (!cuda_alloc_and_copy(problem_sizes_device, problem_sizes_host) ||
      !cuda_alloc_and_copy(ptr_a_device, ptr_a_host) ||
      !cuda_alloc_and_copy(ptr_b_device, ptr_b_host) ||
      !cuda_alloc_and_copy(ptr_sfa_device, ptr_sfa_host) ||
      !cuda_alloc_and_copy(ptr_sfb_device, ptr_sfb_host) ||
      !cuda_alloc_and_copy(ptr_c_device, ptr_c_host) ||
      !cuda_alloc_and_copy(ptr_d_device, ptr_d_host) ||
      !cuda_alloc_and_copy(stride_a_device, stride_a_host) ||
      !cuda_alloc_and_copy(stride_b_device, stride_b_host) ||
      !cuda_alloc_and_copy(stride_c_device, stride_c_host) ||
      !cuda_alloc_and_copy(stride_d_device, stride_d_host) ||
      !cuda_alloc_and_copy(layout_sfa_device, layout_sfa_host) ||
      !cuda_alloc_and_copy(layout_sfb_device, layout_sfb_host)) {
    cleanup();
    return torch::Tensor();
  }

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          device_id);

  typename NativeGemm1::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_groups, problem_sizes_device, problem_sizes_host.data()},
      {ptr_a_device, stride_a_device, ptr_b_device, stride_b_device,
       ptr_sfa_device, layout_sfa_device, ptr_sfb_device,
       layout_sfb_device},
      {{}, ptr_c_device, stride_c_device, ptr_d_device, stride_d_device},
      hw_info};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;

  NativeGemm1 gemm_op;
  auto workspace_size = NativeGemm1::get_workspace_size(args);
  if (workspace_size > 0 &&
      cudaMalloc(reinterpret_cast<void**>(&workspace_device),
                 workspace_size) != cudaSuccess) {
    cleanup();
    return torch::Tensor();
  }

  status = gemm_op.can_implement(args);
  if (status == cutlass::Status::kSuccess) {
    status = gemm_op.initialize(args, workspace_device, stream);
  }
  if (status == cutlass::Status::kSuccess) {
    status = gemm_op.run(stream);
  }
  if (status != cutlass::Status::kSuccess) {
    cleanup();
    return torch::Tensor();
  }

  cleanup();
  return out;
}

// ── GEMM2 SM100 FP8 launch ─────────────────────────────────────────────────

torch::Tensor run_gemm2_grouped_cutlass_fp8(
    const torch::Tensor& a_fp8_padded,
    const torch::Tensor& a_scale_padded,
    const torch::Tensor& w2_fp8_active,
    const torch::Tensor& w2_scale_active,
    const std::vector<std::pair<int64_t, int64_t>>& segments) {
  auto num_groups = static_cast<int>(segments.size());
  if (num_groups == 0) {
    return torch::zeros(
        {0, a_fp8_padded.size(1), w2_fp8_active.size(1)},
        a_scale_padded.options().dtype(torch::kFloat32));
  }

  if (!a_fp8_padded.is_cuda() || !a_scale_padded.is_cuda() ||
      !w2_fp8_active.is_cuda() || !w2_scale_active.is_cuda()) {
    return torch::Tensor();
  }
  if (!a_fp8_padded.is_contiguous() || !a_scale_padded.is_contiguous() ||
      !w2_fp8_active.is_contiguous() || !w2_scale_active.is_contiguous()) {
    return torch::Tensor();
  }
  if (!is_fp8_e4m3_tensor(a_fp8_padded) ||
      !is_fp8_e4m3_tensor(w2_fp8_active)) {
    return torch::Tensor();
  }

  auto max_count = a_fp8_padded.size(1);
  auto i_dim = a_fp8_padded.size(2);
  auto h_dim = w2_fp8_active.size(1);

  if (i_dim % BLOCK != 0 || h_dim % BLOCK != 0) {
    return torch::Tensor();
  }

  auto opts_f32 = torch::TensorOptions()
                      .dtype(torch::kFloat32)
                      .device(a_fp8_padded.device());
  auto out = torch::zeros({num_groups, max_count, h_dim}, opts_f32);

  std::vector<NativeGemm2UnderlyingProblemShape> problem_sizes_host;
  std::vector<const NativeGemm2ElementA*> ptr_a_host;
  std::vector<const NativeGemm2ElementB*> ptr_b_host;
  std::vector<const NativeGemm2ElementAccumulator*> ptr_sfa_host;
  std::vector<const NativeGemm2ElementAccumulator*> ptr_sfb_host;
  std::vector<const float*> ptr_c_host;
  std::vector<float*> ptr_d_host;
  std::vector<NativeGemm2StrideA> stride_a_host;
  std::vector<NativeGemm2StrideB> stride_b_host;
  std::vector<NativeGemm2StrideC> stride_c_host;
  std::vector<NativeGemm2StrideD> stride_d_host;
  std::vector<NativeGemm2LayoutSFA> layout_sfa_host;
  std::vector<NativeGemm2LayoutSFB> layout_sfb_host;
  std::vector<torch::Tensor> packed_sfa_tensors;
  std::vector<torch::Tensor> packed_sfb_tensors;

  problem_sizes_host.reserve(num_groups);
  ptr_a_host.reserve(num_groups);
  ptr_b_host.reserve(num_groups);
  ptr_sfa_host.reserve(num_groups);
  ptr_sfb_host.reserve(num_groups);
  ptr_c_host.reserve(num_groups);
  ptr_d_host.reserve(num_groups);
  stride_a_host.reserve(num_groups);
  stride_b_host.reserve(num_groups);
  stride_c_host.reserve(num_groups);
  stride_d_host.reserve(num_groups);
  layout_sfa_host.reserve(num_groups);
  layout_sfb_host.reserve(num_groups);
  packed_sfa_tensors.reserve(num_groups);
  packed_sfb_tensors.reserve(num_groups);

  auto max_count_m = static_cast<int>(a_fp8_padded.size(1));

  for (int g = 0; g < num_groups; ++g) {
    auto m = max_count_m;
    auto n = static_cast<int>(h_dim);
    auto k = static_cast<int>(i_dim);
    auto logical_k_blocks = static_cast<int64_t>(k) / BLOCK;
    auto logical_n_blocks = static_cast<int64_t>(n) / BLOCK;

    if (m < 16) {
      return torch::Tensor();
    }

    auto layout_sfa = NativeGemm2ScaleConfig::tile_atom_to_shape_SFA(
        cute::make_shape(m, n, k, 1));
    auto layout_sfb = NativeGemm2ScaleConfig::tile_atom_to_shape_SFB(
        cute::make_shape(m, n, k, 1));

    // Pack scales into CUTLASS SM100 expected layout (same as GEMM1)
    auto packed_sfa = pack_block_scale_tensor(
        a_scale_padded[g].slice(0, 0, m),
        layout_sfa, m, logical_k_blocks, /*row_granularity=*/1);
    auto packed_sfb = pack_block_scale_tensor(
        w2_scale_active[g],
        layout_sfb, logical_n_blocks, logical_k_blocks,
        /*row_granularity=*/BLOCK);

    problem_sizes_host.push_back({m, n, k});
    ptr_a_host.push_back(reinterpret_cast<const NativeGemm2ElementA*>(
        a_fp8_padded[g].data_ptr()));
    ptr_b_host.push_back(reinterpret_cast<const NativeGemm2ElementB*>(
        w2_fp8_active[g].data_ptr()));
    packed_sfa_tensors.push_back(std::move(packed_sfa));
    packed_sfb_tensors.push_back(std::move(packed_sfb));
    ptr_sfa_host.push_back(packed_sfa_tensors.back().data_ptr<float>());
    ptr_sfb_host.push_back(packed_sfb_tensors.back().data_ptr<float>());
    ptr_c_host.push_back(out[g].data_ptr<float>());
    ptr_d_host.push_back(out[g].data_ptr<float>());

    stride_a_host.push_back(NativeGemm2StrideA{
        static_cast<int64_t>(k), cute::Int<1>{}, cute::Int<0>{}});
    stride_b_host.push_back(NativeGemm2StrideB{
        static_cast<int64_t>(k), cute::Int<1>{}, cute::Int<0>{}});
    stride_c_host.push_back(NativeGemm2StrideC{
        static_cast<int64_t>(n), cute::Int<1>{}, cute::Int<0>{}});
    stride_d_host.push_back(NativeGemm2StrideD{
        static_cast<int64_t>(n), cute::Int<1>{}, cute::Int<0>{}});
    layout_sfa_host.push_back(layout_sfa);
    layout_sfb_host.push_back(layout_sfb);
  }

  if (problem_sizes_host.empty()) {
    return torch::Tensor();
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cutlass::Status status = cutlass::Status::kSuccess;

  NativeGemm2UnderlyingProblemShape* problem_sizes_device = nullptr;
  const NativeGemm2ElementA** ptr_a_device = nullptr;
  const NativeGemm2ElementB** ptr_b_device = nullptr;
  const NativeGemm2ElementAccumulator** ptr_sfa_device = nullptr;
  const NativeGemm2ElementAccumulator** ptr_sfb_device = nullptr;
  const float** ptr_c_device = nullptr;
  float** ptr_d_device = nullptr;
  NativeGemm2StrideA* stride_a_device = nullptr;
  NativeGemm2StrideB* stride_b_device = nullptr;
  NativeGemm2StrideC* stride_c_device = nullptr;
  NativeGemm2StrideD* stride_d_device = nullptr;
  NativeGemm2LayoutSFA* layout_sfa_device = nullptr;
  NativeGemm2LayoutSFB* layout_sfb_device = nullptr;
  uint8_t* workspace_device = nullptr;

  auto cleanup = [&]() {
    if (workspace_device) cudaFree(workspace_device);
    if (layout_sfb_device) cudaFree(layout_sfb_device);
    if (layout_sfa_device) cudaFree(layout_sfa_device);
    if (stride_d_device) cudaFree(stride_d_device);
    if (stride_c_device) cudaFree(stride_c_device);
    if (stride_b_device) cudaFree(stride_b_device);
    if (stride_a_device) cudaFree(stride_a_device);
    if (ptr_d_device) cudaFree(ptr_d_device);
    if (ptr_c_device) cudaFree(ptr_c_device);
    if (ptr_sfb_device) cudaFree(ptr_sfb_device);
    if (ptr_sfa_device) cudaFree(ptr_sfa_device);
    if (ptr_b_device) cudaFree(ptr_b_device);
    if (ptr_a_device) cudaFree(ptr_a_device);
    if (problem_sizes_device) cudaFree(problem_sizes_device);
  };

  if (!cuda_alloc_and_copy(problem_sizes_device, problem_sizes_host) ||
      !cuda_alloc_and_copy(ptr_a_device, ptr_a_host) ||
      !cuda_alloc_and_copy(ptr_b_device, ptr_b_host) ||
      !cuda_alloc_and_copy(ptr_sfa_device, ptr_sfa_host) ||
      !cuda_alloc_and_copy(ptr_sfb_device, ptr_sfb_host) ||
      !cuda_alloc_and_copy(ptr_c_device, ptr_c_host) ||
      !cuda_alloc_and_copy(ptr_d_device, ptr_d_host) ||
      !cuda_alloc_and_copy(stride_a_device, stride_a_host) ||
      !cuda_alloc_and_copy(stride_b_device, stride_b_host) ||
      !cuda_alloc_and_copy(stride_c_device, stride_c_host) ||
      !cuda_alloc_and_copy(stride_d_device, stride_d_host) ||
      !cuda_alloc_and_copy(layout_sfa_device, layout_sfa_host) ||
      !cuda_alloc_and_copy(layout_sfb_device, layout_sfb_host)) {
    cleanup();
    return torch::Tensor();
  }

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          device_id);

  auto effective_groups = static_cast<int>(problem_sizes_host.size());
  typename NativeGemm2::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {effective_groups, problem_sizes_device, problem_sizes_host.data()},
      {ptr_a_device, stride_a_device, ptr_b_device, stride_b_device,
       ptr_sfa_device, layout_sfa_device, ptr_sfb_device,
       layout_sfb_device},
      {{}, ptr_c_device, stride_c_device, ptr_d_device, stride_d_device},
      hw_info};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;

  NativeGemm2 gemm_op;
  auto workspace_size_val = NativeGemm2::get_workspace_size(args);
  if (workspace_size_val > 0 &&
      cudaMalloc(reinterpret_cast<void**>(&workspace_device),
                 workspace_size_val) != cudaSuccess) {
    cleanup();
    return torch::Tensor();
  }

  status = gemm_op.can_implement(args);
  if (status == cutlass::Status::kSuccess) {
    status = gemm_op.initialize(args, workspace_device, stream);
  }
  if (status == cutlass::Status::kSuccess) {
    status = gemm_op.run(stream);
  }
  if (status != cutlass::Status::kSuccess) {
    cleanup();
    return torch::Tensor();
  }

  cleanup();
  return out;
}

#else  // !FLASHINFER_HAS_CUTLASS_SM100

torch::Tensor run_gemm1_grouped_cutlass_fp8(
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const std::vector<GroupedGemmProblem>&) {
  return torch::Tensor();
}

torch::Tensor run_gemm2_grouped_cutlass_fp8(
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const std::vector<std::pair<int64_t, int64_t>>&) {
  return torch::Tensor();
}

#endif  // FLASHINFER_HAS_CUTLASS_SM100

// ── Custom GEMM2: FP16×FP16→FP32 with tile dispatch & per-k-block scales ──
// Matches Triton's _gemm2_kernel: pre-normalized FP16 A × FP8→FP16 B

// FP8 E4M3 → float conversion that works with __CUDA_NO_HALF_CONVERSIONS__
__device__ __forceinline__ float fp8e4m3_to_float(unsigned char raw) {
  __half_raw hr = __nv_cvt_fp8_to_halfraw(raw, __NV_E4M3);
  // Rebias FP16 bits to FP32 directly to avoid __half operator issues
  unsigned short bits = hr.x;
  unsigned int sign = (bits & 0x8000u) << 16;
  unsigned int exponent = (bits >> 10) & 0x1Fu;
  unsigned int mantissa = bits & 0x03FFu;
  if (exponent == 0) {
    // Zero or denormal FP16 → zero FP32 (good enough for our precision)
    return __uint_as_float(sign);
  }
  // Rebias exponent: FP16 bias=15, FP32 bias=127
  unsigned int f32_bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  return __uint_as_float(f32_bits);
}

// FP16 (__half) → float that works with __CUDA_NO_HALF_CONVERSIONS__
__device__ __forceinline__ float half_to_float(__half val) {
  unsigned short bits;
  memcpy(&bits, &val, sizeof(bits));
  unsigned int sign = (bits & 0x8000u) << 16;
  unsigned int exponent = (bits >> 10) & 0x1Fu;
  unsigned int mantissa = bits & 0x03FFu;
  if (exponent == 0) return __uint_as_float(sign);
  unsigned int f32_bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  return __uint_as_float(f32_bits);
}

constexpr int G2_BLOCK_M = 64;
constexpr int G2_BLOCK_N = 64;
constexpr int G2_BLOCK_K = 128;
constexpr int G2_TM = 4;  // output rows per thread
constexpr int G2_TN = 4;  // output cols per thread
// Thread layout: (16, 16) = 256 threads
// Coverage: 16*4=64 M, 16*4=64 N

// (deleted: gemm2_smem_kernel v1 had scale application bug)
#if 0
__global__ void gemm2_smem_kernel_DELETED(
    const __half* __restrict__ A,          // [tv, K] FP16 pre-normalized
    const float*  __restrict__ A_sc,       // [tv, K_sc] FP32
    const unsigned char* __restrict__ B,   // [E_local, N, K] FP8 as bytes
    const float*  __restrict__ B_sc,       // [E_local, N_sc, K_sc] FP32
    float*        __restrict__ C,          // [tv, N] FP32
    const int*    __restrict__ tile_expert,
    const int*    __restrict__ tile_row,
    const int*    __restrict__ exp_off,
    const int*    __restrict__ exp_cnt,
    const int*    __restrict__ exp_eid,
    int N, int K, int K_sc, int N_sc,
    int stride_be, int stride_bscE) {

  const int pid_m = blockIdx.x;
  const int pid_n = blockIdx.y;

  const int expert_slot = tile_expert[pid_m];
  const int m_block     = tile_row[pid_m];
  const int base_off    = exp_off[expert_slot];
  const int count       = exp_cnt[expert_slot];
  const int eid         = exp_eid[expert_slot];

  const int m_start = m_block * G2_BLOCK_M;
  const int n_start = pid_n * G2_BLOCK_N;

  const int tx = threadIdx.x;  // 0..15 → N dimension
  const int ty = threadIdx.y;  // 0..15 → M dimension
  const int tid = ty * 16 + tx;

  // Shared memory: A tile [BLOCK_M, BLOCK_K] as float, B tile [BLOCK_N, BLOCK_K] as float
  // We convert FP8/FP16 to float during the cooperative load
  __shared__ float smem_A[G2_BLOCK_M][G2_BLOCK_K];
  __shared__ float smem_B[G2_BLOCK_N][G2_BLOCK_K];

  float acc[G2_TM][G2_TN];
  #pragma unroll
  for (int i = 0; i < G2_TM; i++)
    #pragma unroll
    for (int j = 0; j < G2_TN; j++)
      acc[i][j] = 0.0f;

  const unsigned char* B_expert = B + eid * stride_be;
  const float* Bsc_expert = B_sc + eid * stride_bscE;

  const int num_k_blks = K / G2_BLOCK_K;

  for (int k_blk = 0; k_blk < num_k_blks; k_blk++) {
    int k_base = k_blk * G2_BLOCK_K;

    // Cooperative load A [64, 128] FP16 → float into shared memory
    // 256 threads, 64*128 = 8192 elements → 32 per thread
    for (int idx = tid; idx < G2_BLOCK_M * G2_BLOCK_K; idx += 256) {
      int row = idx / G2_BLOCK_K;
      int col = idx % G2_BLOCK_K;
      int m_idx = m_start + row;
      int global_row = base_off + m_idx;
      if (m_idx < count) {
        smem_A[row][col] = half_to_float(A[global_row * K + k_base + col]);
      } else {
        smem_A[row][col] = 0.0f;
      }
    }

    // Cooperative load B [64, 128] FP8 → float into shared memory
    // 256 threads, 64*128 = 8192 elements → 32 per thread
    for (int idx = tid; idx < G2_BLOCK_N * G2_BLOCK_K; idx += 256) {
      int row = idx / G2_BLOCK_K;
      int col = idx % G2_BLOCK_K;
      int n_idx = n_start + row;
      if (n_idx < N) {
        smem_B[row][col] = fp8e4m3_to_float(B_expert[n_idx * K + k_base + col]);
      } else {
        smem_B[row][col] = 0.0f;
      }
    }

    __syncthreads();

    // Load A and B scales for this k_blk
    float a_sc_vals[G2_TM];
    #pragma unroll
    for (int mi = 0; mi < G2_TM; mi++) {
      int m_idx = m_start + ty * G2_TM + mi;
      int global_row = base_off + m_idx;
      a_sc_vals[mi] = (m_idx < count) ? A_sc[global_row * K_sc + k_blk] : 0.0f;
    }

    float b_sc_vals[G2_TN];
    #pragma unroll
    for (int ni = 0; ni < G2_TN; ni++) {
      int n_idx = n_start + tx * G2_TN + ni;
      int n_block = n_idx / 128;
      b_sc_vals[ni] = (n_idx < N) ? Bsc_expert[n_block * K_sc + k_blk] : 0.0f;
    }

    // Compute: each thread accumulates G2_TM × G2_TN output elements
    #pragma unroll
    for (int kk = 0; kk < G2_BLOCK_K; kk++) {
      float a_vals[G2_TM];
      #pragma unroll
      for (int mi = 0; mi < G2_TM; mi++)
        a_vals[mi] = smem_A_flat[(ty * G2_TM + mi) * G2_BLOCK_K + kk];

      float b_vals[G2_TN];
      #pragma unroll
      for (int ni = 0; ni < G2_TN; ni++)
        b_vals[ni] = smem_B_flat[(tx * G2_TN + ni) * G2_BLOCK_K + kk];

      #pragma unroll
      for (int mi = 0; mi < G2_TM; mi++)
        #pragma unroll
        for (int ni = 0; ni < G2_TN; ni++)
          acc[mi][ni] += a_vals[mi] * b_vals[ni];
    }

    // Apply per-k-block scales
    #pragma unroll
    for (int mi = 0; mi < G2_TM; mi++) {
      float combined = a_sc_vals[mi];
      #pragma unroll
      for (int ni = 0; ni < G2_TN; ni++)
        acc[mi][ni] *= combined * b_sc_vals[ni];  // Wrong: scales should multiply dot, not accumulate
    }

    __syncthreads();
  }

}
#endif

// Corrected shared-memory tiled GEMM2
__global__ void gemm2_smem_v2_kernel(
    const __half* __restrict__ A,
    const float*  __restrict__ A_sc,
    const unsigned char* __restrict__ B,
    const float*  __restrict__ B_sc,
    float*        __restrict__ C,
    const int*    __restrict__ tile_expert,
    const int*    __restrict__ tile_row,
    const int*    __restrict__ exp_off,
    const int*    __restrict__ exp_cnt,
    const int*    __restrict__ exp_eid,
    int N, int K, int K_sc, int N_sc,
    int stride_be, int stride_bscE) {

  const int pid_m = blockIdx.x;
  const int pid_n = blockIdx.y;

  const int expert_slot = tile_expert[pid_m];
  const int m_block     = tile_row[pid_m];
  const int base_off    = exp_off[expert_slot];
  const int count       = exp_cnt[expert_slot];
  const int eid         = exp_eid[expert_slot];

  const int m_start = m_block * G2_BLOCK_M;
  const int n_start = pid_n * G2_BLOCK_N;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * 16 + tx;

  // Dynamic shared memory: A [64*128] + B [64*128] floats = 64KB
  extern __shared__ float smem_raw[];
  float* smem_A_flat = smem_raw;
  float* smem_B_flat = smem_raw + G2_BLOCK_M * G2_BLOCK_K;

  float acc[G2_TM][G2_TN];
  #pragma unroll
  for (int i = 0; i < G2_TM; i++)
    #pragma unroll
    for (int j = 0; j < G2_TN; j++)
      acc[i][j] = 0.0f;

  const unsigned char* B_expert = B + eid * stride_be;
  const float* Bsc_expert = B_sc + eid * stride_bscE;

  for (int k_blk = 0; k_blk < K / G2_BLOCK_K; k_blk++) {
    int k_base = k_blk * G2_BLOCK_K;

    // Cooperative load A [64, 128] FP16 → float
    for (int idx = tid; idx < G2_BLOCK_M * G2_BLOCK_K; idx += 256) {
      int row = idx / G2_BLOCK_K;
      int col = idx % G2_BLOCK_K;
      int m_idx = m_start + row;
      int global_row = base_off + m_idx;
      smem_A_flat[row * G2_BLOCK_K + col] = (m_idx < count)
          ? half_to_float(A[global_row * K + k_base + col])
          : 0.0f;
    }

    // Cooperative load B [64, 128] FP8 → float
    for (int idx = tid; idx < G2_BLOCK_N * G2_BLOCK_K; idx += 256) {
      int row = idx / G2_BLOCK_K;
      int col = idx % G2_BLOCK_K;
      int n_idx = n_start + row;
      smem_B_flat[row * G2_BLOCK_K + col] = (n_idx < N)
          ? fp8e4m3_to_float(B_expert[n_idx * K + k_base + col])
          : 0.0f;
    }

    __syncthreads();

    // Compute dot products for this k_blk
    float dot[G2_TM][G2_TN];
    #pragma unroll
    for (int i = 0; i < G2_TM; i++)
      #pragma unroll
      for (int j = 0; j < G2_TN; j++)
        dot[i][j] = 0.0f;

    #pragma unroll
    for (int kk = 0; kk < G2_BLOCK_K; kk++) {
      float a_vals[G2_TM];
      #pragma unroll
      for (int mi = 0; mi < G2_TM; mi++)
        a_vals[mi] = smem_A_flat[(ty * G2_TM + mi) * G2_BLOCK_K + kk];

      float b_vals[G2_TN];
      #pragma unroll
      for (int ni = 0; ni < G2_TN; ni++)
        b_vals[ni] = smem_B_flat[(tx * G2_TN + ni) * G2_BLOCK_K + kk];

      #pragma unroll
      for (int mi = 0; mi < G2_TM; mi++)
        #pragma unroll
        for (int ni = 0; ni < G2_TN; ni++)
          dot[mi][ni] += a_vals[mi] * b_vals[ni];
    }

    // Apply per-k-block scales and accumulate
    #pragma unroll
    for (int mi = 0; mi < G2_TM; mi++) {
      int m_idx = m_start + ty * G2_TM + mi;
      int global_row = base_off + m_idx;
      float a_sc = (m_idx < count) ? A_sc[global_row * K_sc + k_blk] : 0.0f;
      #pragma unroll
      for (int ni = 0; ni < G2_TN; ni++) {
        int n_idx = n_start + tx * G2_TN + ni;
        int n_block = n_idx / 128;
        float b_sc = (n_idx < N) ? Bsc_expert[n_block * K_sc + k_blk] : 0.0f;
        acc[mi][ni] += dot[mi][ni] * a_sc * b_sc;
      }
    }

    __syncthreads();
  }

  // Store output
  #pragma unroll
  for (int mi = 0; mi < G2_TM; mi++) {
    int m_idx = m_start + ty * G2_TM + mi;
    if (m_idx >= count) continue;
    int global_row = base_off + m_idx;
    #pragma unroll
    for (int ni = 0; ni < G2_TN; ni++) {
      int n_idx = n_start + tx * G2_TN + ni;
      if (n_idx >= N)  continue;
      C[global_row * N + n_idx] = acc[mi][ni];
    }
  }
}

// Original naive GEMM2 (kept as fallback reference)
__global__ void gemm2_naive_kernel(
    const __half* __restrict__ A,          // [tv, K] FP16 pre-normalized
    const float*  __restrict__ A_sc,       // [tv, K_sc] FP32
    const __nv_fp8_e4m3* __restrict__ B,   // [E_local, N, K] FP8
    const float*  __restrict__ B_sc,       // [E_local, N_sc, K_sc] FP32
    float*        __restrict__ C,          // [tv, N] FP32
    const int*    __restrict__ tile_expert,
    const int*    __restrict__ tile_row,
    const int*    __restrict__ exp_off,
    const int*    __restrict__ exp_cnt,
    const int*    __restrict__ exp_eid,
    int N, int K, int K_sc, int N_sc,
    int stride_be, int stride_bscE) {

  const int pid_m = blockIdx.x;
  const int pid_n = blockIdx.y;

  const int expert_slot = tile_expert[pid_m];
  const int m_block     = tile_row[pid_m];
  const int base_off    = exp_off[expert_slot];
  const int count       = exp_cnt[expert_slot];
  const int eid         = exp_eid[expert_slot];

  // Each block handles BLOCK_M rows × BLOCK_N cols
  // threadIdx.x covers the N dimension, threadIdx.y covers M dimension
  // Block dim: (32, 8) = 256 threads
  const int local_n = threadIdx.x;  // 0..31
  const int local_m = threadIdx.y;  // 0..7

  // Each thread handles multiple M and N elements
  constexpr int TILE_M = 8;   // rows per thread iteration
  constexpr int TILE_N = 4;   // cols per thread iteration
  // Total coverage: 8 threads * 8 rows = 64 M, 32 threads * 4 cols = 128 N

  const int m_start = m_block * G2_BLOCK_M;
  const int n_start = pid_n * G2_BLOCK_N;

  float acc[TILE_M][TILE_N];
  #pragma unroll
  for (int i = 0; i < TILE_M; i++)
    #pragma unroll
    for (int j = 0; j < TILE_N; j++)
      acc[i][j] = 0.0f;

  const __nv_fp8_e4m3* B_expert = B + eid * stride_be;
  const float* Bsc_expert = B_sc + eid * stride_bscE;

  for (int k_blk = 0; k_blk < K / G2_BLOCK_K; k_blk++) {
    int k_base = k_blk * G2_BLOCK_K;

    // B scale for this (n_block=pid_n, k_block=k_blk) — but per N_sc block
    // B_sc is [E_local, N_sc, K_sc], N_sc = N/128

    #pragma unroll
    for (int mi = 0; mi < TILE_M; mi++) {
      int m_idx = m_start + local_m * TILE_M + mi;
      if (m_idx >= count) continue;
      int global_row = base_off + m_idx;
      float a_sc_val = A_sc[global_row * K_sc + k_blk];

      #pragma unroll
      for (int ni = 0; ni < TILE_N; ni++) {
        int n_idx = n_start + local_n * TILE_N + ni;
        if (n_idx >= N) continue;
        int n_block = n_idx / 128;
        float b_sc_val = Bsc_expert[n_block * K_sc + k_blk];

        float dot = 0.0f;
        const __half* A_row = A + global_row * K + k_base;
        const unsigned char* B_row = reinterpret_cast<const unsigned char*>(
            B_expert) + n_idx * K + k_base;
        #pragma unroll
        for (int kk = 0; kk < G2_BLOCK_K; kk++) {
          float a_f = half_to_float(A_row[kk]);
          float b_f = fp8e4m3_to_float(B_row[kk]);
          dot += a_f * b_f;
        }
        acc[mi][ni] += dot * a_sc_val * b_sc_val;
      }
    }
  }

  // Store output
  #pragma unroll
  for (int mi = 0; mi < TILE_M; mi++) {
    int m_idx = m_start + local_m * TILE_M + mi;
    if (m_idx >= count) continue;
    int global_row = base_off + m_idx;
    #pragma unroll
    for (int ni = 0; ni < TILE_N; ni++) {
      int n_idx = n_start + local_n * TILE_N + ni;
      if (n_idx >= N) continue;
      C[global_row * N + n_idx] = acc[mi][ni];
    }
  }
}

// Host wrapper for GEMM2 tile-dispatch kernel
void run_gemm2_tiled(
    const torch::Tensor& A_fp16,     // [tv, K=2048] FP16
    const torch::Tensor& A_sc,       // [tv, K_sc=16] FP32
    const torch::Tensor& B_fp8,      // [E_local, N=7168, K=2048] FP8
    const torch::Tensor& B_sc,       // [E_local, N_sc=56, K_sc=16] FP32
    torch::Tensor& C,                // [tv, N=7168] FP32 output
    const std::vector<std::pair<int64_t, int64_t>>& segments,
    const std::vector<int64_t>& seg_offsets,
    torch::Device device) {

  int64_t N = B_fp8.size(1);
  int64_t K = B_fp8.size(2);
  int64_t K_sc = K / BLOCK;
  int64_t N_sc = N / BLOCK;
  int64_t num_active = static_cast<int64_t>(segments.size());

  // Build tile dispatch map
  std::vector<int32_t> tile_expert_vec, tile_row_vec;
  std::vector<int32_t> exp_off_vec, exp_cnt_vec, exp_eid_vec;

  for (int64_t k = 0; k < num_active; k++) {
    int64_t cnt = segments[k].second;
    int64_t n_tiles = (cnt + G2_BLOCK_M - 1) / G2_BLOCK_M;
    for (int64_t t = 0; t < n_tiles; t++) {
      tile_expert_vec.push_back(static_cast<int32_t>(k));
      tile_row_vec.push_back(static_cast<int32_t>(t));
    }
    exp_off_vec.push_back(static_cast<int32_t>(seg_offsets[k]));
    exp_cnt_vec.push_back(static_cast<int32_t>(cnt));
    exp_eid_vec.push_back(static_cast<int32_t>(segments[k].first));
  }

  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto tile_expert_t = torch::from_blob(
      tile_expert_vec.data(), {static_cast<int64_t>(tile_expert_vec.size())},
      torch::kInt32).to(device);
  auto tile_row_t = torch::from_blob(
      tile_row_vec.data(), {static_cast<int64_t>(tile_row_vec.size())},
      torch::kInt32).to(device);
  auto exp_off_t = torch::from_blob(
      exp_off_vec.data(), {num_active}, torch::kInt32).to(device);
  auto exp_cnt_t = torch::from_blob(
      exp_cnt_vec.data(), {num_active}, torch::kInt32).to(device);
  auto exp_eid_t = torch::from_blob(
      exp_eid_vec.data(), {num_active}, torch::kInt32).to(device);

  int num_m_tiles = static_cast<int>(tile_expert_vec.size());
  int grid_n = (static_cast<int>(N) + G2_BLOCK_N - 1) / G2_BLOCK_N;

  dim3 grid(num_m_tiles, grid_n);
  dim3 block(32, 8);  // 256 threads: 32 along N, 8 along M
  auto stream = at::cuda::getCurrentCUDAStream();

  gemm2_naive_kernel<<<grid, block, 0, stream.stream()>>>(
      reinterpret_cast<const __half*>(A_fp16.data_ptr()),
      A_sc.data_ptr<float>(),
      reinterpret_cast<const __nv_fp8_e4m3*>(B_fp8.data_ptr()),
      B_sc.data_ptr<float>(),
      C.data_ptr<float>(),
      tile_expert_t.data_ptr<int32_t>(),
      tile_row_t.data_ptr<int32_t>(),
      exp_off_t.data_ptr<int32_t>(),
      exp_cnt_t.data_ptr<int32_t>(),
      exp_eid_t.data_ptr<int32_t>(),
      static_cast<int>(N), static_cast<int>(K),
      static_cast<int>(K_sc), static_cast<int>(N_sc),
      static_cast<int>(B_fp8.stride(0)),
      static_cast<int>(N_sc * K_sc));
}

// ── SwiGLU ──────────────────────────────────────────────────────────────────

__global__ void swiglu_kernel(
    const float* __restrict__ g1,
    float* __restrict__ out,
    int64_t rows,
    int64_t i) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * i;
  if (idx >= total) {
    return;
  }

  int64_t row = idx / i;
  int64_t col = idx - row * i;
  int64_t g1_row = row * (2 * i);

  float up = g1[g1_row + col];
  float gate = g1[g1_row + i + col];
  float sig = 1.0f / (1.0f + expf(-gate));
  out[idx] = up * gate * sig;
}

torch::Tensor apply_swiglu(const torch::Tensor& g1, int64_t i) {
  if (g1.is_cuda() && g1.scalar_type() == torch::kFloat32 &&
      g1.is_contiguous()) {
    auto rows = g1.numel() / (2 * i);
    auto out_shape = g1.sizes().vec();
    out_shape.back() = i;
    auto out = torch::empty(out_shape, g1.options());

    constexpr int threads = 256;
    int64_t total = rows * i;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    swiglu_kernel<<<blocks, threads, 0, stream.stream()>>>(
        g1.data_ptr<float>(), out.data_ptr<float>(), rows, i);
    return out;
  }

  auto up = g1.slice(g1.dim() - 1, 0, i);
  auto gate = g1.slice(g1.dim() - 1, i, 2 * i);
  return (gate * torch::sigmoid(gate)) * up;
}

// ── FP8 Block-Scale Quantization ────────────────────────────────────────────

std::tuple<torch::Tensor, torch::Tensor> quantize_to_fp8_blockwise(
    const torch::Tensor& input) {
  auto tv = input.size(0);
  auto dim = input.size(1);
  auto k_blocks = dim / BLOCK;

  auto blk = input.reshape({tv, k_blocks, BLOCK});
  auto amax = blk.abs().amax(-1).clamp_min(1e-12f);
  auto scale = amax / 448.0f;
  auto normalized = (blk / scale.unsqueeze(-1)).clamp(-448.0f, 448.0f);
  auto fp8 =
      normalized.reshape({tv, dim}).to(torch::kFloat8_e4m3fn).contiguous();

  return {fp8, scale.contiguous()};
}

// ── FP32 Dequant Fallbacks ──────────────────────────────────────────────────

torch::Tensor dequant_activation_grouped(
    const torch::Tensor& fp8,
    const torch::Tensor& scale) {
  auto G = fp8.size(0), M = fp8.size(1), K = fp8.size(2);
  auto kb = K / BLOCK;
  auto blk = fp8.to(torch::kFloat32).reshape({G, M, kb, BLOCK});
  return (blk * scale.unsqueeze(-1)).reshape({G, M, K});
}

torch::Tensor dequant_weight_grouped(
    const torch::Tensor& fp8,
    const torch::Tensor& scale) {
  auto G = fp8.size(0), N = fp8.size(1), K = fp8.size(2);
  auto nb = N / BLOCK, kb = K / BLOCK;
  auto blk = fp8.to(torch::kFloat32)
                 .reshape({G, nb, BLOCK, kb, BLOCK})
                 .permute({0, 1, 3, 2, 4});
  return (blk * scale.unsqueeze(-1).unsqueeze(-1))
      .permute({0, 1, 3, 2, 4})
      .reshape({G, N, K});
}

// ── Routing ─────────────────────────────────────────────────────────────────

std::tuple<torch::Tensor, torch::Tensor> deepseek_v3_routing(
    const torch::Tensor& routing_logits,
    const torch::Tensor& routing_bias,
    double routed_scaling_factor) {
  auto t = routing_logits.size(0);
  auto device = routing_logits.device();
  auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(device);

  auto logits = routing_logits.to(torch::kFloat32);
  auto bias = routing_bias.to(torch::kFloat32).reshape({-1});

  auto s = torch::sigmoid(logits);
  auto s_with_bias = s + bias;

  auto group_size = E_GLOBAL / N_GROUP;
  auto s_wb_grouped = s_with_bias.view({t, N_GROUP, group_size});

  auto top2_vals = std::get<0>(s_wb_grouped.topk(2, 2, true, false));
  auto group_scores = top2_vals.sum(2);

  auto group_idx = std::get<1>(group_scores.topk(TOPK_GROUP, 1, true, false));
  auto group_mask = torch::zeros({t, N_GROUP}, opts_f);
  group_mask.scatter_(1, group_idx, 1.0);

  auto score_mask = group_mask.unsqueeze(2)
                        .expand({t, N_GROUP, group_size})
                        .reshape({t, E_GLOBAL});

  auto neg_inf = -std::numeric_limits<float>::infinity();
  auto scores_pruned = s_with_bias.masked_fill(score_mask.eq(0), neg_inf);
  auto topk_idx =
      std::get<1>(scores_pruned.topk(TOP_K, 1, true, false));

  auto topk_scores = s.gather(1, topk_idx);
  auto weights_sum = topk_scores.sum(1, true) + 1e-20;
  auto topk_weights =
      (topk_scores / weights_sum) * routed_scaling_factor;

  return {topk_idx.contiguous(), topk_weights.contiguous()};
}

}  // namespace

// ── Kernel entry point (DPS) ────────────────────────────────────────────────

torch::Tensor kernel(
    const torch::Tensor& routing_logits,
    const torch::Tensor& routing_bias,
    const torch::Tensor& hidden_states,
    const torch::Tensor& hidden_states_scale,
    const torch::Tensor& gemm1_weights,
    const torch::Tensor& gemm1_weights_scale,
    const torch::Tensor& gemm2_weights,
    const torch::Tensor& gemm2_weights_scale,
    py::object local_expert_offset,
    py::object routed_scaling_factor,
    torch::Tensor output) {
  auto device = hidden_states.device();
  auto t = routing_logits.size(0);
  auto e_local = gemm1_weights.size(0);
  auto h = hidden_states.size(1);
  auto i = gemm2_weights.size(2);
  auto gemm1_out = gemm1_weights.size(1);  // 2 * I

  auto local_start = to_int64(local_expert_offset);
  auto scaling = to_double(routed_scaling_factor);

  auto opts_f32 =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  // ── Step 1: Routing ────────────────────────────────────────────────────────
  auto routing = deepseek_v3_routing(routing_logits, routing_bias, scaling);
  auto topk_idx = std::get<0>(routing);
  auto topk_weights = std::get<1>(routing);

  // ── Step 2: Sort tokens by expert ──────────────────────────────────────────
  auto token_ids =
      torch::arange(t, torch::TensorOptions().dtype(torch::kLong).device(device))
          .unsqueeze(1)
          .expand({t, TOP_K})
          .reshape({-1});
  auto expert_ids = topk_idx.reshape({-1});
  auto flat_weights = topk_weights.reshape({-1});
  auto local_eids = expert_ids - local_start;

  auto valid = local_eids.ge(0).__and__(local_eids.lt(e_local));
  auto v_tok = token_ids.masked_select(valid);
  auto v_eid = local_eids.masked_select(valid);
  auto v_wt = flat_weights.masked_select(valid);

  if (v_tok.numel() == 0) {
    output.zero_();
    return output;
  }

  auto sort_result = v_eid.sort();
  auto sorted_eids = std::get<0>(sort_result);
  auto sort_indices = std::get<1>(sort_result);
  auto sorted_tids = v_tok.index_select(0, sort_indices);
  auto sorted_weights = v_wt.index_select(0, sort_indices);

  // Gather FP8 activations and scales (sorted by expert)
  auto a_fp8_sorted =
      hidden_states.index_select(0, sorted_tids).contiguous();
  auto a_sc_sorted = hidden_states_scale.transpose(0, 1)
                         .index_select(0, sorted_tids)
                         .contiguous();

  // Build per-expert segments
  auto expert_counts_cpu =
      torch::bincount(sorted_eids, torch::Tensor(), e_local)
          .to(torch::kCPU, torch::kLong);
  auto counts_ptr = expert_counts_cpu.data_ptr<int64_t>();
  int64_t tv = sorted_eids.numel();

  std::vector<std::pair<int64_t, int64_t>> segments;
  for (int64_t le = 0; le < e_local; ++le) {
    if (counts_ptr[le] > 0) {
      segments.emplace_back(le, counts_ptr[le]);
    }
  }
  int64_t num_active = static_cast<int64_t>(segments.size());

  std::vector<int64_t> seg_offsets(num_active + 1, 0);
  for (int64_t k = 0; k < num_active; ++k) {
    seg_offsets[k + 1] = seg_offsets[k] + segments[k].second;
  }

  int64_t max_count = 0;
  for (auto& seg : segments) {
    max_count = std::max(max_count, seg.second);
  }
  // Pad to multiple of 128 for SM100 1-SM schedule
  int64_t max_count_padded = ((max_count + 127) / 128) * 128;

  std::vector<int64_t> active_ids;
  active_ids.reserve(num_active);
  for (auto& seg : segments) {
    active_ids.push_back(seg.first);
  }
  auto active_ids_t = torch::tensor(
      active_ids,
      torch::TensorOptions().dtype(torch::kLong).device(device));

  // ── Step 3: Pad FP8 activations into grouped format ────────────────────────
  auto a_fp8_padded = torch::zeros(
      {num_active, max_count_padded, h},
      torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device));
  auto a_sc_padded = torch::zeros(
      {num_active, max_count_padded, h / BLOCK}, opts_f32);

  for (int64_t k = 0; k < num_active; ++k) {
    int64_t off = seg_offsets[k];
    int64_t cnt = segments[k].second;
    a_fp8_padded[k].slice(0, 0, cnt).copy_(
        a_fp8_sorted.slice(0, off, off + cnt));
    a_sc_padded[k].slice(0, 0, cnt).copy_(
        a_sc_sorted.slice(0, off, off + cnt));
  }

  auto w1_fp8_active =
      gemm1_weights.index_select(0, active_ids_t).contiguous();
  auto w1_sc_active =
      gemm1_weights_scale.index_select(0, active_ids_t).contiguous();

  // ── Step 4: GEMM1 — FP8 grouped CUTLASS SM100 ─────────────────────────────
  std::vector<GroupedGemmProblem> gemm1_problems;
  gemm1_problems.reserve(num_active);
  for (int64_t k = 0; k < num_active; ++k) {
    gemm1_problems.push_back({k, max_count_padded, gemm1_out, h});
  }

  auto g1_out = run_gemm1_grouped_cutlass_fp8(
      a_fp8_padded, a_sc_padded, w1_fp8_active, w1_sc_active,
      gemm1_problems);

  if (!g1_out.defined()) {
    auto a_dequant = dequant_activation_grouped(a_fp8_padded, a_sc_padded);
    auto w1_dequant = dequant_weight_grouped(w1_fp8_active, w1_sc_active);
    g1_out = torch::bmm(a_dequant,
                         w1_dequant.transpose(1, 2).contiguous());
  }

  // ── Step 5: SwiGLU on padded output ────────────────────────────────────────
  auto tv_padded = num_active * max_count_padded;
  auto c_out = apply_swiglu(g1_out.reshape({tv_padded, gemm1_out}), i)
                   .reshape({num_active, max_count_padded, i});

  // ── Step 6: GEMM2 — Custom FP16×FP16→FP32 with tile dispatch ────────────
  // Unpad SwiGLU output to flat [tv, I]
  auto c_flat = torch::empty({tv, i}, opts_f32);
  for (int64_t k = 0; k < num_active; ++k) {
    int64_t off = seg_offsets[k];
    int64_t cnt = segments[k].second;
    c_flat.slice(0, off, off + cnt).copy_(
        c_out[k].slice(0, 0, cnt));
  }

  // Pre-normalize to FP16 per 128-block (matching Triton)
  auto c_blk = c_flat.reshape({tv, i / BLOCK, BLOCK});
  auto c_amax = c_blk.abs().amax(-1).clamp_min(1e-8f);  // [tv, I/128]
  auto c_fp16 = (c_blk / c_amax.unsqueeze(-1))
                    .reshape({tv, i})
                    .to(torch::kFloat16)
                    .contiguous();
  auto c_sc = c_amax.contiguous();  // [tv, I/128] FP32

  // Launch custom GEMM2 kernel — no weight dequant, no padding
  auto o_flat = torch::zeros({tv, h}, opts_f32);
  run_gemm2_tiled(
      c_fp16, c_sc,
      gemm2_weights, gemm2_weights_scale,
      o_flat, segments, seg_offsets, device);

  auto result = torch::zeros({t, h}, opts_f32);
  result.index_add_(0, sorted_tids,
                    o_flat * sorted_weights.unsqueeze(1));

  output.copy_(result.to(torch::kBFloat16));
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kernel", &kernel, "MoE kernel (CUDA extension)");
}
