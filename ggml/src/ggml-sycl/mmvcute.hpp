//
// MIT license
// Copyright (C) 2025 Codeplay Software Ltd.
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_MMVCUTE_HPP
#define GGML_SYCL_MMVCUTE_HPP

#include "common.hpp"

typedef float (*vec_dot_cute_sycl_t)(const void * __restrict__ vbq, const ggml_half *   d4s,
                                     const block_q8_1 * __restrict__ bq8_1, const uint32_t iqs, const uint32_t base_iq_index, const uint32_t ncols, const uint32_t nrows, const uint32_t row);

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

namespace ggml_sycl_reordered {

template <ggml_type type> struct block_q_t;

// Expected memory layout / sizes
// ncols * nrows / QK4_0          blocks and scales
// ncols * nrows * QK4_0          quants
// ncols * nrows * QK4_0 / QR4_0  bytes
template <> struct block_q_t<GGML_TYPE_Q4_0> {
    struct traits {
        static constexpr uint32_t qk       = QK4_0;
        static constexpr uint32_t qi       = QI4_0;
        static constexpr uint32_t qr       = QR4_0;
        static constexpr uint32_t vdr_mmvq = 2;
    };

    // qs and d are contiguous in memory, out-of-bounds qs will access to d values
    uint8_t *   qs;
    ggml_half * d;
};

}  // namespace ggml_sycl_reordered

void ggml_sycl_op_mul_mat_vec_cute(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                   ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                   const char * src1_ddq_i, float * dst_dd_i, int64_t row_low, int64_t row_high,
                                   int64_t src1_ncols, int64_t src1_padded_col_size, const dpct::queue_ptr & stream);

#endif  // GGML_SYCL_MMVCUTE_HPP
