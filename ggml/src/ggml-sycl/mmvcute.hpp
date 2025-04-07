//
// MIT license
// Copyright (C) 2025 Codeplay Software Ltd.
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_MMVCUTE_HPP
#define GGML_SYCL_MMVCUTE_HPP

#include "common.hpp"

typedef float (*vec_dot_cute_sycl_t)(const void * __restrict__ vbq,
                                     const int & iqs, const int ibx_offset, const int d_offset, const block_q8_1 * __restrict__ bq8_1, const int * iby,const int & iqs2, const int tid, const int i,
                                     const size_t row, bool print);

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// Defined by how the vec dot algorithm is written
constexpr size_t VDR_Q6_K_Q8_1_MMVCUTE = 1;
constexpr size_t VDR_Q4_K_Q8_1_MMVCUTE = 2;

#ifdef __SYCL_DEVICE_ONLY__
template <class T, int N> using vector_t = T __attribute__((ext_vector_type(N)));
#else
template <class T, int N> using vector_t = sycl::marray<T, N>;
#endif

#ifdef __SYCL_DEVICE_ONLY__
namespace detail {
SYCL_EXTERNAL extern "C" vector_t<ushort, 2> __builtin_IB_subgroup_block_read_flat_u8_m1k32v2(
    long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, vector_t<int, 2> coord);
SYCL_EXTERNAL extern "C" uint32_t __builtin_IB_subgroup_block_read_flat_u32_m1k16v1(
    long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, vector_t<int, 2> coord);
}  // namespace detail
#endif

namespace quants {
namespace reordered {
namespace detail {

struct block_q4_0_traits {};

}  // namespace detail

// Expected
// ncols * nrows / QK4_0          blocks and scales
// ncols * nrows * QK4_0          quants
// ncols * nrows * QK4_0 / QR4_0  bytes
struct block_q4_0 {
    struct traits {
        static constexpr size_t qk       = QK4_0;
        static constexpr size_t qi       = QI4_0;
        static constexpr size_t qr       = QR4_0;
        static constexpr size_t vdr_q8_1 = 2;
    };

    // qs and d are contiguous in memory, out-of-bounds qs will access to d values
    uint8_t *   qs;
    ggml_half * d;
};

}  // namespace reordered
}  // namespace quants

void ggml_sycl_op_mul_mat_vec_cute(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                   ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                   const char * src1_ddq_i, float * dst_dd_i, int64_t row_low, int64_t row_high,
                                   int64_t src1_ncols, int64_t src1_padded_col_size, const dpct::queue_ptr & stream);


// template <int qk, int qi, typename block_q_t, int vdr, vec_dot_cute_sycl_t vec_dot_cute_sycl /* , class Tensor_VX */
//           /*, class TensorVY, class TensorDST */>
// static void mul_mat_vec_cute(/* Tensor_VX tensor_vx, */ const void * vx, /* TensorVY */ const void * vy,
//                              float * /* TensorDST */ dst, const int ncols, const int nrows,
//                              const sycl::nd_item<3> & nd_item) {
//     const int row = nd_item.get_group(2) * nd_item.get_local_range(1) + nd_item.get_local_id(1);
//
//     if (row >= nrows) {
//         return;
//     }
//
//     const int     blocks_per_row      = ncols / qk;
//     constexpr int blocks_per_subgroup = safe_div(vdr * WARP_SIZE, qi);  // Ensuring blocks_per_subgroup > 0
//
//     constexpr int block_elements_per_subgroup = qi / vdr;
//
//     assert(blocks_per_subgroup > 0);
//     assert(block_elements_per_subgroup > 0);
//
//     // partial sum for each thread
//     float partial_sum = 0.0f;
//
//     // using Element_VX        = typename Tensor_VX::value_type;
//     // using Copy_Thread_Shape = cute::Shape<cute::Int<WARP_SIZE>, cute::_1>;
//     // using Traits_Load_VX    = cute::Copy_Traits<cute::XE_2D_U8x1x32_LD_N, Tensor_VX>;
//     // using Atom_Load_VX      = cute::Copy_Atom<Traits_Load_VX, Element_VX>;
//
//     // auto tiled_copy_load_vx =
//     //     make_tiled_copy(Atom_Load_VX{}.with(tensor_vx), cute::Layout<Copy_Thread_Shape>{},
//     //                     cute::make_layout(cute::shape_div(typename Traits_Load_VX::BlockShape{}, Copy_Thread_Shape{})));
//
//     // auto         subgroup               = nd_item.get_sub_group();
//     // auto         first_thread_in_sg_idx = subgroup.get_group_linear_id() * WARP_SIZE;
//     // cute::Tensor tCgA                   = thr_mma.partition_A(gA);
//
//     // auto         thr_copy_A = tiled_copy_load_vx.get_slice(nd_item.get_local_id(2));
//     // cute::Tensor tVX =
//     //     cute::make_tensor<Element_VX>(cute::make_shape(cute::C<Atom_Load_VX::NumValDst>{}, cute::_1{}, cute::_1{}));
//
//     // auto blk_load_s = cute::get_pvc_tensor(tensor_vx.shape());
//     // auto thread_s   = thr_copy_A.partition_S(blk_load_s(cute::_, cute::_, 0));
//
//     // clang-format off
// #if 0
//     // if (cute::thread0()) {
//     //     cute::print("tensor_vx: "); cute::print(tensor_vx); cute::print("\n");
//     //     cute::print("thr_copy_A: "); cute::print(thr_copy_A); cute::print("\n");
//     //     // cute::print("cute_tensor_vy: "); cute::print(cute_tensor_vy); cute::print("\n");
//     //     // cute::print("cute_tensor_dst: "); cute::print(cute_tensor_dst); cute::print("\n");
//     //     cute::print("thr_copy_A: "); cute::print(thr_copy_A); cute::print("\n");
//     //     // cute::print("blk_load_s: "); cute::print(blk_load_s); cute::print("\n");
//     //     // cute::print("thread_s: "); cute::print(thread_s); cute::print("\n");
//     //     cute::print("\n\n");
//     // }
//     syncthreads();
//     // if (cute::thread(1 * WARP_SIZE)) {
//     //     cute::print("tensor_vx: "); cute::print(tensor_vx); cute::print("\n");
//     //     cute::print("thr_copy_A: "); cute::print(thr_copy_A); cute::print("\n");
//     //     // cute::print("cute_tensor_vy: "); cute::print(cute_tensor_vy); cute::print("\n");
//     //     // cute::print("cute_tensor_dst: "); cute::print(cute_tensor_dst); cute::print("\n");
//     // }
// #endif
//     // clang-format on
//
//     const block_q_t *  x = (const block_q_t *) vx;
//     const block_q8_1 * y = (const block_q8_1 *) vy;
//
//     // Prefetch 0, 1 from B two iby[values]
//     // Prefetch 0, 1 from A two ibx[] values
//
//     for (size_t i = nd_item.get_local_id(2) / block_elements_per_subgroup; i < blocks_per_row;
//          i += blocks_per_subgroup) {
//         const int ibx = row * blocks_per_row + i;  // x block index
//
//         const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx
//
//         for (size_t elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
//             // x block quant index when casting the quants to int
//             const int iqs = elem + vdr * (nd_item.get_local_id(2) % block_elements_per_subgroup);
//
//             partial_sum += vec_dot_cute_sycl(&x[ibx], &y[iby], iqs);
//
//             if (nd_item.get_local_id(2) < 0) {
//                 cute::print(
//                     "mul_mat_vec_cute: tid=%d item<2>=%d row=%d, bpr=%d bps=%d eps=%d i=%d elem=%d ibx=%d iby=%d "
//                     "iqs=%d,partial_sum=%.8f\n",
//                     nd_item.get_global_linear_id(), nd_item.get_local_id(2), row, blocks_per_row, blocks_per_subgroup,
//                     block_elements_per_subgroup, i, elem, ibx, iby, iqs, partial_sum);
//             }
//         }
//     }
//
//     auto sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum, std::plus<>());
//
//     if (nd_item.get_local_id(2) == 0) {
//         dst[row] = sum;
//     }
// }


#endif  // GGML_SYCL_MMVCUTE_HPP
