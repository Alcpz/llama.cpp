//
// MIT license
// Copyright (C) 2025 Codeplay Software Ltd.
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "mmvcute.hpp"

#include "dpct/helper.hpp"

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

static __dpct_inline__ int get_int_from_uint8(const uint8_t * x8, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) (x8 + sizeof(int) * i32);  // assume at least 2 byte
                                                                         // alignment

    int x32 = 0;
    x32 |= x16[0] << 0;
    x32 |= x16[1] << 16;

    return x32;
}

static __dpct_inline__ int get_int_from_int8_aligned(const int8_t * x8, const int & i32) {
    return *((const int *) (x8 + sizeof(int) * i32));  // assume at least 4 byte alignment
}

constexpr size_t safe_div(const size_t m, const size_t n) {
    assert(n > 0);
    return (m + n - 1) / n;
}

template <ggml_type q_t, vec_dot_cute_sycl_t vec_dot_cute_sycl, size_t pipeline_prefetch = 2, size_t rows_per_sg = 1>
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
    // Prefetch de 0..1
    for (size_t num_prefetch = 0; num_prefetch < pipeline_prefetch; num_prefetch++) {
        // TODO
    }
    for (size_t row = base_row; row < (base_row + rows_per_sg) && row < nrows; row++) {
        float partial_sum = 0.0f;

        const size_t scales_per_row = ncols / block_traits::qk;
        const size_t d_offset       = (ncols / block_traits::qr * nrows) + scales_per_row * row * sizeof(ggml_half);

        for (size_t i = sg_local_id / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
            // prefetch 2 + i
            const size_t base_iq_index = (i / blocks_per_subgroup) * (block_traits::vdr_mmvq * WARP_SIZE);

            // if (ThreadIdxX() % 256 == 0 || ThreadIdxX() % 256 == 16 || ThreadIdxX() % 256 == 32) {
            //     cute::print("\n - wid=%d, tid=%d d_offset=%zu", workgroup_id, nd_item.get_global_id(2), (d_offset - (ncols / block_traits::qr * nrows)) / sizeof(ggml_half));
            // }

            const ggml_half * d4s = reinterpret_cast<const ggml_half *>(static_cast<const uint8_t *>(vx) + d_offset);

            for (size_t elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
                // block quant index of the row when casting the quants to int
                const int iqs = elem + (sg_local_id % block_traits::qi);

                partial_sum += vec_dot_cute_sycl(x, d4s, y, iqs, base_iq_index, ncols, nrows, row);
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
static __dpct_inline__ float vec_dot_q4_0_q8_1([[maybe_unused]] const void * __restrict__ vbq, const ggml_half * d4s,
                                               const block_q8_1 * __restrict__ bq8_1, const size_t               iqs,
                                               const size_t base_iq_index, [[maybe_unused]] const size_t ncols,
                                               [[maybe_unused]] const size_t nrows, [[maybe_unused]] const size_t row) {
    using sycl::ext::oneapi::this_work_item::get_sub_group;
    using q4_0_traits                    = typename ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_0>::traits;
    constexpr size_t blocks_per_subgroup = safe_div(q4_0_traits::vdr_mmvq * WARP_SIZE, q4_0_traits::qi);
    const int        global_id           = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_global_id(2);
    const int        workgroup_id        = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group_linear_id();
    const int        local_id            = get_sub_group().get_local_linear_id();
    const int        sg_id               = get_sub_group().get_group_linear_id();

    float dot = 0.f;

#pragma unroll
    for (size_t q = 0; q < q4_0_traits::vdr_mmvq; q++) {
        uint v = 0;

        const size_t     load_sg_index = base_iq_index + (q * WARP_SIZE);
        vector_t<int, 2> coord         = { load_sg_index, row };

#ifdef __SYCL_DEVICE_ONLY__
        size_t width  = ncols / (q4_0_traits::qr);
        size_t height = nrows;  // INFO: nrows

        *reinterpret_cast<uint *>(&v) = detail::__builtin_IB_subgroup_block_read_flat_u32_m1k16v1(
            (long) (vbq), width - 1, height - 1, width - 1, coord);
#endif

        if (ThreadIdxX() == 0) {
            // cute::print("\n - cols=%zu, rows=%zu, scales_per_row=%d", ncols, nrows, (ncols / q4_0_traits::qk));
        }

        const size_t    iq_index = base_iq_index + (q * WARP_SIZE + local_id);
        const int       ibq8_1   = (iq_index / q4_0_traits::qi) * (q4_0_traits::qk / QK8_1);
        // blocks per subgroup  == blocks in a row
        // const size_t    d_index  = row * (ncols / q4_0_traits::qk) + (iq_index / (blocks_per_subgroup / 2));
        // const ggml_half d4       = d4s[d_index];
        const ggml_half d4       = d4s[iq_index / (blocks_per_subgroup / 2)];
        const auto &    b        = bq8_1[ibq8_1];
        const int       u0       = get_int_from_int8_aligned(b.qs, iqs);
        const int       u1       = get_int_from_int8_aligned(b.qs, iqs + q4_0_traits::qi);

        if (ThreadIdxX() % 256 == 0 || ThreadIdxX() % 256 == 16) {
            // cute::print("\n - wid=%d, tid=%d coord {%zu, %zu} -> iq_index=%d, ibq8_1=%d v=%d", workgroup_id, global_id,
            //             load_sg_index, row, iq_index, ibq8_1, v);
        }

        dot += vec_dot_q4_0_q8_1_impl(v, u0, u1, d4, b.ds);
    }

    return dot;
}

// NOTE: Will only work with GGML_SYCL_DISABLE_OPT=1 for now
static void mul_mat_vec_cute_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                            const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    // TODO: What is the purpose of GGML_SYCL_MMV_Y
    const int        block_num_y       = safe_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t cute_sg_per_wg    = 16;
    constexpr size_t rows_per_sg       = 1;
    constexpr size_t pipeline_prefetch = 2;
    GGML_ASSERT(block_num_y % cute_sg_per_wg == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, (block_num_y * WARP_SIZE) / rows_per_sg);
    const sycl::range<3> wg_size(1, GGML_SYCL_MMV_Y, cute_sg_per_wg * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, wg_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_ocl<GGML_TYPE_Q4_0, vec_dot_q4_0_q8_1, pipeline_prefetch, rows_per_sg>(
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
            case GGML_TYPE_Q4_0:
                mul_mat_vec_cute_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            default:
                GGML_ABORT("Unsupported quantization reached in mmvcute");
        }
    }
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(ctx);
}
