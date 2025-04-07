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
template <size_t rows_per_sg, int qk, int qi, int qr, typename block_q_t, int vdr,
          vec_dot_cute_sycl_t vec_dot_cute_sycl>
static void mul_mat_vec_ocl(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                            const size_t ncols, const size_t nrows, const sycl::nd_item<3> & nd_item) {
    const auto   sg           = nd_item.get_sub_group();
    const int    sg_range     = sg.get_group_linear_range();
    const int    workgroup_id = nd_item.get_group_linear_id();
    const int    sg_id        = sg.get_group_linear_id();
    const int    sg_global_id = workgroup_id * sg_range + sg_id;
    // TODO: Considering more than one sg per row maybe?
    // constexpr size_t sgs_per_row  = 1;
    // const size_t     base_row     = rows_per_sg * safe_div(sg_global_id, sgs_per_row);
    const size_t base_row     = rows_per_sg * sg_global_id;

    if (base_row >= nrows) {
        return;
    }

    const size_t     blocks_per_row              = ncols / qk;
    constexpr size_t blocks_per_subgroup         = safe_div(vdr * WARP_SIZE, qi);  // Ensuring blocks_per_subgroup > 0
    constexpr size_t block_elements_per_subgroup = qi / vdr;

    assert(blocks_per_subgroup > 0);
    assert(block_elements_per_subgroup > 0);

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    // TODO: Change this for the actual base_row instead of an offset
    for (size_t row = base_row; row < (base_row + rows_per_sg) && row < nrows; row++) {
        float partial_sum = 0.0f;
        for (size_t i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row;
             i += blocks_per_subgroup) {
            const size_t ibx        = row * blocks_per_row + i;  // x block index
            // TODO: Generalize offset. Right now only works for quantizations that don't split
            // in high/low bits (3,5,6)
            const size_t ibx_offset = ibx * (qk / qr);
            // ncols / qr -> size of weights in number of bytes
            const size_t scales_per_row = ncols / QK4_0;
            const size_t d_offset  = (ncols / qr * nrows) + scales_per_row * row * sizeof(ggml_half);

            // TODO: Change to baseptr + internally calculated offset maybe?
            // TODO: Reorder q8_1?
            int iby[vdr + 1];
            iby[0] = i * (qk / QK8_1);
            for (size_t q8blocki = 0; q8blocki < vdr; q8blocki++) {
                // Which integer of quants is loaded on X
                // Row not needed because each row is independent
                size_t quant_coord =
                    (i / blocks_per_subgroup) * (vdr * WARP_SIZE) + (sg.get_local_linear_id() + q8blocki * WARP_SIZE);
                iby[q8blocki + 1] = (quant_coord / qi) * (qk / QK8_1);
            }

            bool print = false;
            for (size_t elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
                // x block quant index when casting the quants to int
                const int iqs  = elem + vdr * (sg.get_local_linear_id() % block_elements_per_subgroup);
                const int iqs2 = elem + (sg.get_local_linear_id() % qi);

                // clang-format off
                if (sg_global_id == 2
                        // && nd_item.get_local_id(2) % 4 == 0
                        // && i == sg.get_local_linear_id() / block_elements_per_subgroup
                        ) {
                    print = true;
                    // INFO: to have better ganularity on what to print
                    for (size_t j = 0; j < nd_item.get_global_range(2); j++) {
                        if (cute::thread(j)) {
                            if (cute::thread0()) {
                                cute::print("\n ==============================================");
                                cute::print("\nmul_mat_vec_ocl: blocks_per_sg=%zu", blocks_per_subgroup);
                            }
                            cute::print("\nmul_mat_vec_ocl: ");
                            cute::print("tid=%d ", nd_item.get_global_linear_id());
                            cute::print("item<2>=%d ", nd_item.get_local_id(2));
                            cute::print("local_id<2>=%d ", sg.get_local_linear_id());
                            cute::print("row=%d, ", row);
                            // cute::print("bpr=%d ", blocks_per_row);
                            // cute::print("bps=%d ", blocks_per_subgroup);
                            // cute::print("eps=%d ", block_elements_per_subgroup);
                            cute::print("i=%d ", i);
                            cute::print("elem=%d ", elem);
                            cute::print("ibx=%zu ", ibx);
                            cute::print("ibx_offset=%zu ", ibx_offset);
                            // cute::print("iby=%d ", iby);
                            // cute::print("iby[0]=%d ", iby[0]);
                            // cute::print("iby[1]=%d ", iby[1]);
                            // cute::print("iby[2]=%d ", iby[2]);
                            cute::print("iqs=%d ", iqs);
                            cute::print("iqs2=%d ", iqs2);
                        }
                    }
                }
                // clang-format on
                partial_sum += vec_dot_cute_sycl(x, iqs, ibx_offset, d_offset, y, iby, iqs2,
                                                 sg.get_local_linear_id(), i, row, print);
            }
        }

        auto sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum, std::plus<>());
        if (sg.leader()) {
            dst[row] = sum;
        }
    }
}

static __dpct_inline__ float vec_dot_q4_0_q8_1_impl2(const int * v, const int * u, const float & d4_0,
                                                     const float & d4_1, const sycl::half2 & ds8_0,
                                                     const sycl::half2 & ds8_1) {
    int sumi0 = 0;
    int sumi1 = 0;

    using block_traits = quants::reordered::block_q4_0::traits;
    static_assert(block_traits::vdr_q8_1 == 2, "This implementation assumes VDR = 2");

    const sycl::float2 ds8f0 = ds8_0.convert<float, sycl::rounding_mode::automatic>();
    const sycl::float2 ds8f1 = ds8_1.convert<float, sycl::rounding_mode::automatic>();

#pragma unroll
    for (size_t i = 0; i < block_traits::vdr_q8_1; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // TODO: Branchless
        if (i == 0) {
            sumi0 = dpct::dp4a(vi0, u[2 * i + 0], sumi0);
            sumi0 = dpct::dp4a(vi1, u[2 * i + 1], sumi0);
        } else if (i == 1) {
            sumi1 = dpct::dp4a(vi0, u[2 * i + 0], sumi1);
            sumi1 = dpct::dp4a(vi1, u[2 * i + 1], sumi1);
        }
    }

    float dot0 = sumi0 * ds8f0.x() - ((8 / QI4_0) * ds8f0.y());
    float dot1 = sumi1 * ds8f1.x() - ((8 / QI4_0) * ds8f1.y());

    return d4_0 * dot0 + d4_1 * dot1;
}

static __dpct_inline__ float vec_dot_q4_0_q8_1_impl(const int * v, const int * u, const float & d4,
                                                    const sycl::half2 & ds8) {
    int sumi = 0;

    using block_traits = quants::reordered::block_q4_0::traits;
#pragma unroll
    for (size_t i = 0; i < block_traits::vdr_q8_1; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

    const sycl::float2 ds8f = ds8.convert<float, sycl::rounding_mode::automatic>();

    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x() - (8 * block_traits::vdr_q8_1 / QI4_0) * ds8f.y());
}

static __dpct_inline__ float vec_dot_q4_0_q8_1(const void * __restrict__ vbq, const int & iqs, const int ibx_offset,
                                               const int d_offset,
                                               const block_q8_1 * __restrict__ bq8_1, const int * iby, const int & iqs2,
                                               [[maybe_unused]] const int tid, [[maybe_unused]] const int i,
                                               [[maybe_unused]] const size_t row, [[maybe_unused]] bool print) {
    using block_traits                      = quants::reordered::block_q4_0::traits;
    constexpr size_t VDR_Q4_0_Q8_1_MMVCUTE = block_traits::vdr_q8_1;

#ifdef __SYCL_DEVICE_ONLY__
    constexpr size_t blocks_per_subgroup = safe_div(block_traits::vdr_q8_1 * WARP_SIZE, block_traits::qi);
    int              v_block[VDR_Q4_0_Q8_1_MMVCUTE];
    vector_t<int, 2> coord[VDR_Q4_0_Q8_1_MMVCUTE];
    for (size_t iq = 0; iq < block_traits::vdr_q8_1; iq++) {
        coord[iq] = { (i / blocks_per_subgroup) * (block_traits::vdr_q8_1 * WARP_SIZE) + (tid + iq * WARP_SIZE), row };
        size_t width  = 512 * sizeof(uint8_t);  // INFO: Size in bytes of block aka ncols / traits::qi
        size_t height = 16;                     // INFO: nrows

        // *reinterpret_cast<vector_t<ushort, 2> *>(v) =
        // u8 -> atomic value to bytes
        // m1 -> vertical dimension
        // k32 -> k-dimension size (also deifnes the stride of the values received by threads)
        // v2 -> 2 values
        // This translates to 64 bytes loaded, in two pairs of two bytes because:
        //   k32 * v2 = 64 u8 values,
        //   v2 = two blocks loaded 64 / 2 -> 32 u8 values are loaded per block,
        //   32 / WARP_SIZE = 2 values per thread
        // detail::__builtin_IB_subgroup_block_read_flat_u8_m1k32v2((long) (vbq), 512 * sizeof(uint8_t) - 1, 16 - 1,
        //                                                          512 * sizeof(uint8_t) - 1, coord);

        // baseoffset ALWAYS BEGINNING OF MEM REGION
        // Dimensions are defined by builtin (1k16v1) 1 x (16 * 1)
        // width (bytes)     -> actual width in bytes of the memory region
        // height (elements) -> 1
        // pitch (bytes)     -> helps defining the subblock of memory to access
        // x,y coord like system that helps identifying the block to load. Row major.
        *reinterpret_cast<uint *>(&v_block[iq]) = __builtin_IB_subgroup_block_read_flat_u32_m1k16v1(
            (long) (vbq), width - 1, height - 1, width - 1, coord[iq]);
    }
#endif

    const uint8_t * bq4_0 = static_cast<const uint8_t *>(vbq) + ibx_offset;
    int             v[VDR_Q4_0_Q8_1_MMVCUTE];
    int             u[2 * VDR_Q4_0_Q8_1_MMVCUTE];   // q8 bytes == 2 * q4 bytes
    int             u2[2 * VDR_Q4_0_Q8_1_MMVCUTE];  // q8 bytes == 2 * q4 bytes

#pragma unroll
    for (size_t i = 0; i < VDR_Q4_0_Q8_1_MMVCUTE; ++i) {
        v[i]         = get_int_from_uint8(bq4_0, iqs + i);                        // A
        u[2 * i + 0] = get_int_from_int8_aligned((&bq8_1[iby[0]])->qs, iqs + i);  // B
        u[2 * i + 1] = get_int_from_int8_aligned((&bq8_1[iby[0]])->qs, iqs + i + QI4_0);
    }
    for (size_t i = 0; i < VDR_Q4_0_Q8_1_MMVCUTE; ++i) {
        u2[2 * i + 0] = get_int_from_int8_aligned((&bq8_1[iby[i + 1]])->qs, iqs2);  // B
        u2[2 * i + 1] = get_int_from_int8_aligned((&bq8_1[iby[i + 1]])->qs, iqs2 + QI4_0);
    }

#ifdef __SYCL_DEVICE_ONLY__
    const ggml_half d1 =
        *(reinterpret_cast<const ggml_half *>(static_cast<const uint8_t *>(vbq) + d_offset + sizeof(ggml_half) * (coord[0][0] / 4)));
    const ggml_half d2 =
        *(reinterpret_cast<const ggml_half *>(static_cast<const uint8_t *>(vbq) + d_offset + sizeof(ggml_half) * (coord[1][0] / 4)));
    auto res2 = vec_dot_q4_0_q8_1_impl2(v_block, u2, d1, d2, (&bq8_1[iby[1]])->ds, (&bq8_1[iby[2]])->ds);

    auto sg   = syclcompat::get_nd_item<1>().get_sub_group();
    auto sum2 = sycl::reduce_over_group(sg, res2, std::plus<>());
    if (print) {
        cute::print("\ntid=%d, coord=%d, v_block(%d), d=%.8f", tid, coord[0][0], v_block[0], static_cast<float>(d1));
        cute::print("\ntid=%d, coord=%d, v_block(%d), d=%.8f", tid, coord[1][0], v_block[1], static_cast<float>(d2));
        // cute::print("\ntid=%d, (iqs + i)=%d, (iqs + i + QI4_0)=%d", tid, iqs, iqs + QI4_0);
        // cute::print("\ntid=%d, (iqs2 + i)=%d, (iqs2 + i + QI4_0)=%d", tid, iqs2, iqs2 + QI4_0);
        // cute::print("\ntid=%d, v(%d), u(%d, %d), d=%.8f", tid, v[0], u[0], u[1],
        //             static_cast<float>(d));
        // cute::print("\ntid=%d, v_block(%d), u2(%d, %d), d=%.8f", tid, v_block[0], u2[0], u2[1],
        //             static_cast<float>(d));
        // cute::print("\ntid=%d, v(%d), u(%d, %d), d=%.8f", tid, v[1], u[2], u[3],
        //             static_cast<float>(d));
        // cute::print("\ntid=%d, v_block(%d), u2(%d, %d), d=%.8f", tid, v_block[1], u2[2], u2[3],
        //             static_cast<float>(d));
        // cute::print("\ntid=%d, iby=%d, iqs=%d, v(%d), v_block(%d), u(%d), u2(%d)", tid, iby[0], iqs + 1, v[1],
        // v_block[1], u[2], u2[2]);
    }

    return res2;
#endif

    return 0;
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

    constexpr size_t VDR_Q4_0_Q8_1_MMVCUTE = quants::reordered::block_q4_0::traits::vdr_q8_1;
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(global_size, wg_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                // mul_mat_vec_cute<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVCUTE, vec_dot_q4_0_q8_1>(
                //     /* cute_tensor_vx, */ vx, vy, dst, ncols, nrows, nd_item);

                mul_mat_vec_ocl<rows_per_sg, QK4_0, QI4_0, QR4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVCUTE, vec_dot_q4_0_q8_1>(
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
