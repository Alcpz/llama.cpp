//
// MIT license
// Copyright (C) 2025 Codeplay Software Ltd.
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_MMVCUTE_HPP
#define GGML_SYCL_MMVCUTE_HPP

#include "common.hpp"

typedef float (*vec_dot_cute_sycl_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
                                     const int & iqs, const int ibx_offset, const int d_offset);

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

#endif  // GGML_SYCL_MMVCUTE_HPP
