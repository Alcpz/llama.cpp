//
// MIT license
// Copyright (C) 2025 Codeplay Software Ltd.
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_MMVCUTE_HPP
#define GGML_SYCL_MMVCUTE_HPP

#include "common.hpp"

typedef float (*vec_dot_cute_sycl_t)(const void * __restrict__ vbq, const ggml_half *    d4s,
                                     const block_q8_1 * __restrict__ bq8_1, const size_t iqs,
                                     const size_t base_iq_index, const size_t ncols, const size_t nrows,
                                     const size_t row);

#ifdef __SYCL_DEVICE_ONLY__
template <class T, int N> using vector_t = T __attribute__((ext_vector_type(N)));
#else
template <class T, int N> using vector_t = sycl::marray<T, N>;
#endif

#ifdef __SYCL_DEVICE_ONLY__
#    define __global __attribute__((opencl_global))

namespace detail {
// block_load
SYCL_EXTERNAL extern "C" uint32_t __builtin_IB_subgroup_block_read_flat_u32_m1k16v1(
    long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, vector_t<int, 2> coord);

// prefetch
SYCL_EXTERNAL void intel_sub_group_2d_block_prefetch_32b_1r16x1c(
        const __global void* base_address, int width, int height, int pitch,
        vector_t<int, 2> coord);
}  // namespace detail

#    undef __global
#endif

void ggml_sycl_op_mul_mat_vec_cute(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                   ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                   const char * src1_ddq_i, float * dst_dd_i, int64_t row_low, int64_t row_high,
                                   int64_t src1_ncols, int64_t src1_padded_col_size, const dpct::queue_ptr & stream);

#endif  // GGML_SYCL_MMVCUTE_HPP
