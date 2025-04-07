//
// MIT license
// Copyright (C) 2025 Codeplay Software Ltd.
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "mmvcute.hpp"

#include "dpct/helper.hpp"
#include "ggml.h"

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

// TODO: create a traits struct for qk, qi, qr
template <ggml_type q_t, vec_dot_cute_sycl_t vec_dot_cute_sycl, size_t rows_per_sg = 1>
static void mul_mat_vec_ocl(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                            const size_t ncols, const size_t nrows, const sycl::nd_item<3> & nd_item) {
    using block_traits = typename ggml_sycl_reordered::block_q_t<q_t>::traits;
    using block_type   = ggml_sycl_reordered::block_q_t<q_t>;

    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  sg_local_id  = sg.get_local_linear_id();
    const int  sg_global_id = workgroup_id * sg_range + sg_id;

    const size_t base_row = rows_per_sg * sg_global_id;
    if (base_row >= nrows) {
        return;
    }

    const size_t     blocks_per_row = ncols / block_traits::qk;
    constexpr size_t blocks_per_subgroup =
        safe_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);  // Ensuring blocks_per_subgroup > 0
    constexpr size_t block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;

    assert(blocks_per_subgroup > 0);
    assert(block_elements_per_subgroup > 0);

    const block_type * x = (const block_type *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    for (size_t row = base_row; row < (base_row + rows_per_sg) && row < nrows; row++) {
        float partial_sum = 0.0f;
        for (size_t i = sg_local_id / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
            const size_t scales_per_row = ncols / block_traits::qk;
            // ncols / block_traits::qr -> bytes of quants
            const size_t d_offset       = (ncols / block_traits::qr * nrows) + scales_per_row * row * sizeof(ggml_half);

            // TODO: Change to baseptr + internally calculated offset, prefetch + load.
            // TODO: Reorder q8_1?
            int iby[block_traits::vdr_mmvq];
            for (size_t q = 0; q < block_traits::vdr_mmvq; q++) {
                // quant block (an int) that is loaded on X
                // Row not needed because each row is independent
                size_t quant_coord =
                    (i / blocks_per_subgroup) * (block_traits::vdr_mmvq * WARP_SIZE) + (sg_local_id + q * WARP_SIZE);
                iby[q] = (quant_coord / block_traits::qi) * (block_traits::qk / QK8_1);
            }

            for (size_t elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
                // block quant index of the row when casting the quants to int
                const int iqs = elem + (sg_local_id % block_traits::qi);

                // clang-format off

                // if (sg_global_id == 0
                //         && nd_item.get_local_id(2) % 4 == 0
                //         && i == sg.get_local_linear_id() / block_elements_per_subgroup
                //         ) {
                //     // INFO: to have better ganularity on what to print
                //     for (size_t j = 0; j < nd_item.get_global_range(2); j++) {
                //         if (cute::thread(j)) {
                //             if (cute::thread0()) {
                //                 cute::print("\n ==============================================");
                //                 cute::print("\nmul_mat_vec_ocl: blocks_per_sg=%zu", blocks_per_subgroup);
                //             }
                //             cute::print("\nmul_mat_vec_ocl: ");
                //             cute::print("tid=%d ", nd_item.get_global_linear_id());
                //             cute::print("item<2>=%d ", nd_item.get_local_id(2));
                //             cute::print("local_id<2>=%d ", sg_local_id);
                //             cute::print("row=%d, ", row);
                //             // cute::print("bpr=%d ", blocks_per_row);
                //             // cute::print("bps=%d ", blocks_per_subgroup);
                //             // cute::print("eps=%d ", block_elements_per_subgroup);
                //             cute::print("i=%d ", i);
                //             cute::print("elem=%d ", elem);
                //             cute::print("ibx=%zu ", ibx);
                //             cute::print("ibx_offset=%zu ", ibx_offset);
                //             // cute::print("iby=%d ", iby);
                //             // cute::print("iby[0]=%d ", iby[0]);
                //             // cute::print("iby[1]=%d ", iby[1]);
                //             // cute::print("iby[2]=%d ", iby[2]);
                //             cute::print("iqs=%d ", iqs);
                //         }
                //     }
                // }

                // clang-format on
                partial_sum += vec_dot_cute_sycl(x, d_offset, y, iby, iqs, sg_local_id, i, row);
            }
        }

        auto sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum, std::plus<>());
        if (sg.leader()) {
            dst[row] = sum;
        }
    }
}

static __dpct_inline__ float vec_dot_q4_0_q8_1_impl(const int v, const int u0, const int u1, const float d4,
                                                    const sycl::half2 & ds8) {
    using q4_0_traits = typename ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_0>::traits;
    static_assert(q4_0_traits::vdr_mmvq == 2, "This implementation assumes VDR = 2");

    const sycl::float2 ds8f = ds8.convert<float, sycl::rounding_mode::automatic>();

    const int vi0 = (v >> 0) & 0x0F0F0F0F;
    const int vi1 = (v >> 4) & 0x0F0F0F0F;

    int sumi = 0;
    sumi     = dpct::dp4a(vi0, u0, sumi);
    sumi     = dpct::dp4a(vi1, u1, sumi);

    float dot = sumi * ds8f.x() - ((8 / q4_0_traits::qi) * ds8f.y());
    return d4 * dot;
}

// TODO: d_offset -> d_ptr to beginning of row
static __dpct_inline__ float vec_dot_q4_0_q8_1(const void * __restrict__ vbq, const int           d_offset,
                                               const block_q8_1 * __restrict__ bq8_1, const int * iby, const int & iqs,
                                               const int tid, const int i, [[maybe_unused]] const size_t row) {
    using q4_0_traits                    = typename ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_0>::traits;
    constexpr size_t blocks_per_subgroup = safe_div(q4_0_traits::vdr_mmvq * WARP_SIZE, q4_0_traits::qi);

    float dot = 0.f;

    const size_t base_col = (i / blocks_per_subgroup) * (q4_0_traits::vdr_mmvq * WARP_SIZE);
#pragma unroll
    for (size_t iq = 0; iq < q4_0_traits::vdr_mmvq; iq++) {
        int v = 0;

        const size_t     col   = base_col + (iq * WARP_SIZE + tid);
        vector_t<int, 2> coord = { col, row };

#ifdef __SYCL_DEVICE_ONLY__
        size_t width  = 512 * sizeof(uint8_t);  // INFO: Size in bytes of block aka ncols / traits::qi
        size_t height = 16;                     // INFO: nrows

        *reinterpret_cast<uint *>(&v) =
            __builtin_IB_subgroup_block_read_flat_u32_m1k16v1((long) (vbq), width - 1, height - 1, width - 1, coord);
#endif

        const ggml_half d4 = *(reinterpret_cast<const ggml_half *>(static_cast<const uint8_t *>(vbq) + d_offset +
                                                                   sizeof(ggml_half) * (col / 4)));

        const auto & b  = bq8_1[iby[iq]];
        const int    u0 = get_int_from_int8_aligned(b.qs, iqs);
        const int    u1 = get_int_from_int8_aligned(b.qs, iqs + q4_0_traits::qi);

        dot += vec_dot_q4_0_q8_1_impl(v, u0, u1, d4, b.ds);
    }

    return dot;
}

// NOTE: Will only work with GGML_SYCL_DISABLE_OPT=1 for now
static void mul_mat_vec_cute_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                            const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    // TODO: What is the purpose of GGML_SYCL_MMV_Y
    const int        block_num_y    = safe_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t cute_sg_per_wg = 16;
    constexpr size_t rows_per_sg    = 1;
    GGML_ASSERT(block_num_y % cute_sg_per_wg == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, (block_num_y * WARP_SIZE) / rows_per_sg);
    const sycl::range<3> wg_size(1, GGML_SYCL_MMV_Y, cute_sg_per_wg * WARP_SIZE);

    // TODO: It may be simpler to switch around the shape and set it to LayoutLeft
    // vx is (ncols, nrows), vy is (ncols, 1), dst (nrows, 1), where dims are 0..1
    // auto vec_dot_shape  = cute::make_shape(1, ncols, nrows);
    // auto cute_tensor_vx = cute::make_tensor(cute::make_gmem_ptr(static_cast<const uint8_t *>(vx)),
    //                                         cute::make_layout(cute::select<1, 2>(vec_dot_shape), cute::LayoutLeft{}));
    // auto cute_tensor_vy = cute::make_tensor(static_cast<const uint8_t *>(vy),
    //                                         cute::make_layout(cute::select<1, 0>(vec_dot_shape), cute::LayoutRight{}));
    // auto cute_tensor_dst =
    //     cute::make_tensor(dst, cute::make_layout(cute::select<2, 0>(vec_dot_shape), cute::LayoutRight{}));
    // printf("nrows=%d, ncols=%d\n", nrows, ncols);
    // printf("global_size=%zu,%zu,%zu, local_size=%zu,%zu,%zu\n", global_size[0], global_size[1], global_size[2],
    //        wg_size[0], wg_size[1], wg_size[2]);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, wg_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             // mul_mat_vec_cute<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVCUTE, vec_dot_q4_0_q8_1>(
                             //     /* cute_tensor_vx, */ vx, vy, dst, ncols, nrows, nd_item);

                             mul_mat_vec_ocl<GGML_TYPE_Q4_0, vec_dot_q4_0_q8_1, rows_per_sg>(vx, vy, dst, ncols, nrows,
                                                                                             nd_item);
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
            case GGML_TYPE_Q4_0:
                mul_mat_vec_cute_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            // case GGML_TYPE_Q4_K:
            //     mul_mat_vec_cute_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
            //     break;
            // case GGML_TYPE_Q6_K:
            //     mul_mat_vec_cute_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
            //     break;
            default:
                GGML_ABORT("Unsupported quantization reached in mmvcute");
        }
    }
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(ctx);
}
