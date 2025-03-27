//
// MIT license
// Copyright (C) 2025 Codeplay Software Ltd.
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "mmvcute.hpp"

#include "cute/arch/xe_copy_1B.hpp"
#include "dpct/helper.hpp"
#include "sycl/group_algorithm.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wunused-local-typedef"

#include <cute/tensor.hpp>
#include <cute/util/debug.hpp>

#pragma clang diagnostic pop

// TODO: Figure this out
static __dpct_inline__ int get_int_from_uint8(const uint8_t * x8, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) (x8 + sizeof(int) * i32);  // assume at least 2 byte
                                                                         // alignment

    int x32 = 0;
    x32 |= x16[0] << 0;
    x32 |= x16[1] << 16;

    return x32;
}

// TODO: Figure this out
static __dpct_inline__ int get_int_from_int8_aligned(const int8_t * x8, const int & i32) {
    return *((const int *) (x8 + sizeof(int) * i32));  // assume at least 4 byte alignment
}

constexpr size_t safe_div(const size_t m, const size_t n) {
    assert(n > 0);
    return (m + n - 1) / n;
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_cute_sycl_t vec_dot_cute_sycl, class TensorVX,
          class TensorVY, class TensorDST>
static void mul_mat_vec_cute(TensorVX vx, TensorVY vy, TensorDST dst, const int ncols, const int nrows,
                             const sycl::nd_item<3> & nd_item) {
    const int row = nd_item.get_group(2) * nd_item.get_local_range(1) + nd_item.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row      = ncols / qk;
    constexpr int blocks_per_subgroup = safe_div(vdr * WARP_SIZE, qi);  // Ensuring blocks_per_subgroup > 0

    constexpr int block_elements_per_subgroup = qi / vdr;

    assert(blocks_per_subgroup > 0);
    assert(block_elements_per_subgroup > 0);

    // partial sum for each thread
    float partial_sum = 0.0f;

    // TODO: It may be simpler to switch around the shape and set it to LayoutLeft
    // vx is (ncols, nrows), vy is (ncols, 1), dst (nrows, 1), where dims are 0..1
    auto vec_dot_shape = cute::make_shape(1, ncols, nrows);

    auto cute_tensor_vx = cute::make_tensor(static_cast<const uint8_t *>(vx),
                                            cute::make_layout(cute::select<1, 2>(vec_dot_shape), cute::LayoutRight{}));
    auto cute_tensor_vy = cute::make_tensor(static_cast<const uint8_t *>(vy),
                                            cute::make_layout(cute::select<1, 0>(vec_dot_shape), cute::LayoutRight{}));
    auto cute_tensor_dst =
        cute::make_tensor(dst, cute::make_layout(cute::select<2, 0>(vec_dot_shape), cute::LayoutRight{}));


    using CopyThreadShape = cute::Shape<cute::_1, cute::Int<WARP_SIZE>>;
    using traits_load = cute::Copy_Traits<cute::XE_2D_U8x1x32_LD_N, decltype(cute_tensor_vx)>;


    if (cute::thread0()) {
        cute::print("cute_tensor_vx: ");
        cute::print(cute_tensor_vx);
        cute::print("\n");
        cute::print("cute_tensor_vy: ");
        cute::print(cute_tensor_vy);
        cute::print("\n");
        cute::print("cute_tensor_dst: ");
        cute::print(cute_tensor_dst);
        cute::print("\n");
    }

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    // Prefetch 0, 1 from B two iby[values]
    // Prefetch 0, 1 from A two ibx[] values

    for (int i = nd_item.get_local_id(2) / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        for (size_t elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            // x block quant index when casting the quants to int
            const int iqs = elem + vdr * (nd_item.get_local_id(2) % block_elements_per_subgroup);

            partial_sum += vec_dot_cute_sycl(&x[ibx], &y[iby], iqs);

            // if (nd_item.get_local_id(2) == 0) {
            //     cute::print(
            //         "mul_mat_vec_cute: tid=%d item<2>=%d row=%d, bpr=%d bps=%d eps=%d i=%d elem=%d ibx=%d iby=%d "
            //         "iqs=%d\n" /* partial_sum=%.8f\n" */,
            //         nd_item.get_global_linear_id(), nd_item.get_local_id(2), row, blocks_per_row, blocks_per_subgroup,
            //         block_elements_per_subgroup, i, elem, ibx, iby, iqs /* , partial_sum*/);
            // }
        }
    }

    auto sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum, std::plus<>());

    if (nd_item.get_local_id(2) == 0) {
        dst[row] = sum;
    }
}

// contiguous v/x values
static __dpct_inline__ float vec_dot_q4_K_q8_1_impl_vmmq(const int * __restrict__ v, const int * __restrict__ u,
                                                         const uint8_t * __restrict__ sc,
                                                         const uint8_t * __restrict__ m, const sycl::half2 & dm4,
                                                         const float * __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;

        const int dot1 = dpct::dp4a(v1i, u[2 * i + 1], dpct::dp4a(v0i, u[2 * i + 0], 0));  // SIMD dot product
        const int dot2 = dpct::dp4a(0x01010101, u[2 * i + 1], dpct::dp4a(0x01010101, u[2 * i + 0], 0));  // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const sycl::float2 dm4f = dm4.convert<float, sycl::rounding_mode::automatic>();

    return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

static __dpct_inline__ float vec_dot_q4_K_q8_1(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
                                               const int & iqs) {
    const block_q4_K * bq4_K = (const block_q4_K *) vbq;

    int   v[2];
    int   u[2 * QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));

    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

    const int * q4 = (const int *) (bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0]           = q4[0];
    v[1]           = q4[4];

    const uint16_t * scales = (const uint16_t *) bq4_K->scales;
    uint16_t         aux[2];
    const int        j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *) aux;
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i]                   = bq8i->ds[0];

        const int * q8 = (const int *) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0]   = q8[0];
        u[2 * i + 1]   = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

static void mul_mat_vec_cute_q4_K_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                            const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int            block_num_y = safe_div(nrows, GGML_SYCL_MMV_Y);
    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> local_size(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, local_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_cute<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVCUTE, vec_dot_q4_K_q8_1>(
                                 vx, vy, dst, ncols, nrows, nd_item);
                         });
    });
}

void ggml_sycl_op_mul_mat_vec_cute(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                   ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                   const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low,
                                   const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_col_size,
                                   const dpct::queue_ptr & stream) {
    constexpr size_t q8_1_ts = sizeof(block_q8_1);
    constexpr size_t q8_1_bs = QK8_1;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % q8_1_bs == 0);

    const int64_t ne00     = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    // TODO: Figure out what this comments means
    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    for (int i = 0; i < src1_ncols; i++) {
        const size_t src1_ddq_i_offset = i * src1_padded_col_size * q8_1_ts / q8_1_bs;
        const char * src1_ddq_i_bs     = src1_ddq_i + src1_ddq_i_offset;
        float *      dst_dd_i_bs       = dst_dd_i + i * dst->ne[0];
        switch (src0->type) {
            case GGML_TYPE_Q4_K:
                mul_mat_vec_cute_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            // case GGML_TYPE_Q6_K:
            //     mul_mat_vec_cute_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
            //     break;
            default:
                GGML_ABORT("fatal error");
        }
    }
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(ctx);
}
