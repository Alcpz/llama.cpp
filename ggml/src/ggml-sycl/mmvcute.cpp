//
// MIT license
// Copyright (C) 2025 Codeplay Software Ltd.
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "mmvcute.hpp"

#include "dpct/helper.hpp"

constexpr size_t safe_div(const size_t m, const size_t n) {
    assert(n > 0);
    return (m + n - 1) / n;
}

// contiguous v/x values
static __dpct_inline__ float vec_dot_q6_K_q8_1_impl_mmvcute(const int & vl, const int & vh, const int * __restrict__ u,
                                                            const int8_t * __restrict__ scales, const float & d,
                                                            const float * __restrict__ d8) {
    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4 * i];

        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;

        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;

        const int vi =
            dpct::vectorized_binary<sycl::char4>((vil | vih), 0x20202020, dpct::sub_sat());  // vi = (vil | vih) - 32

        sumf += d8[i] * (dpct::dp4a(vi, u[i], 0) * sc);                                      // SIMD dot product
    }

    return d * sumf;
}

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

static __dpct_inline__ float vec_dot_cute_q6_K_q8_1(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
                                               const int & iqs) {
    const block_q6_K * bq6_K = (const block_q6_K *) vbq;

    const int bq8_offset   = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
    const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
    const int vh_shift     = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));

    const int vl = get_int_from_uint8(bq6_K->ql, iqs);
    const int vh = get_int_from_uint8(bq6_K->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int   u[QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_offset + 2 * i].ds[0];
    }

    return vec_dot_q6_K_q8_1_impl_mmvcute(vl, vh, u, scales, bq6_K->d, d8);
}




template <int qk, int qi, typename block_q_t, int vdr, vec_dot_cute_sycl_t vec_dot_cute_sycl>
static void mul_mat_vec_cute(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                          const int ncols, const int nrows, const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row      = ncols / qk;
    constexpr int blocks_per_subgroup = safe_div(vdr * WARP_SIZE, qi);  // Ensuring blocks_per_subgroup > 0

    constexpr size_t elements_per_workitem = qi / vdr;

    assert(blocks_per_subgroup > 0);
    assert(elements_per_workitem > 0);

    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        for (size_t elem = 0; elem < elements_per_workitem; elem += WARP_SIZE) {
            // x block quant index when casting the quants to int
            const int iqs = elem + vdr * (item_ct1.get_local_id(2) % elements_per_workitem);

            tmp += vec_dot_cute_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

// static void mul_mat_vec_q4_K_q8_1_sycl(const void *vx, const void *vy,
//                                        float *dst, const int ncols,
//                                        const int nrows,
//                                        dpct::queue_ptr stream) {
//     GGML_ASSERT(ncols % QK_K == 0);
//     const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
//     const sycl::range<3> block_nums(1, 1, block_num_y);
//     const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
//     {
//
//         stream->submit([&](sycl::handler &cgh) {
//
//             cgh.parallel_for(
//                 sycl::nd_range<3>(block_nums * block_dims, block_dims),
//                 [=](sycl::nd_item<3> item_ct1)
//                     [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
//                         mul_mat_vec_q<QK_K, QI4_K, block_q4_K,
//                                       VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
//                             vx, vy, dst, ncols, nrows, item_ct1);
//                     });
//         });
//     }
// }

static void mul_mat_vec_cute_q6_K_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                            const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_cute<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVCUTE, vec_dot_cute_q6_K_q8_1>(
                        vx, vy, dst, ncols, nrows, item_ct1);
                });
        });
    }
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
            // case GGML_TYPE_Q4_K:
            //     mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
            //     break;
            case GGML_TYPE_Q6_K:
                mul_mat_vec_cute_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            default:
                GGML_ABORT("fatal error");
        }
    }
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(ctx);
}
