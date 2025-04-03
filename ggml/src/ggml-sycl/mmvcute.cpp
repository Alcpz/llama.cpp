//
// MIT license
// Copyright (C) 2025 Codeplay Software Ltd.
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "mmvcute.hpp"

#include <cstdio>

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

#ifdef __SYCL_DEVICE_ONLY__
template <class T, int N> using vector_t = T __attribute__((ext_vector_type(N)));
#else
template <class T, int N> using vector_t = sycl::marray<T, N>;
#endif

#ifdef __SYCL_DEVICE_ONLY__
namespace detail {
SYCL_EXTERNAL extern "C" vector_t<ushort, 2> __builtin_IB_subgroup_block_read_flat_u8_m1k32v2(
    long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, vector_t<int, 2> coord);
}
#endif

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

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_cute_sycl_t vec_dot_cute_sycl>
static void mul_mat_vec_ocl(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                            const int ncols, const int nrows, const sycl::nd_item<3> & nd_item) {
    const auto       sg           = nd_item.get_sub_group();
    const int        sg_range     = sg.get_group_linear_range();
    const int        workgroup_id = nd_item.get_group_linear_id();
    const int        sg_id        = sg.get_group_linear_id();
    const int        sg_global_id = workgroup_id * sg_range + sg_id;
    constexpr size_t sgs_per_row  = 1;  // TODO: Adjust appropriately
    const int        row          = safe_div(sg_global_id, sgs_per_row);

    if (row >= nrows) {
        return;
    }

    const size_t     blocks_per_row      = ncols / qk;
    constexpr size_t blocks_per_subgroup = safe_div(vdr * WARP_SIZE, qi);  // Ensuring blocks_per_subgroup > 0

    constexpr size_t block_elements_per_subgroup = qi / vdr;

    assert(blocks_per_subgroup > 0);
    assert(block_elements_per_subgroup > 0);

    // partial sum for each thread
    float partial_sum = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    // TODO: Change this for the actual subgroup id
    for (size_t i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row;
         i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        for (size_t elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            // x block quant index when casting the quants to int
            const int iqs = elem + vdr * (sg.get_local_linear_id() % block_elements_per_subgroup);

            partial_sum += vec_dot_cute_sycl(&x[ibx], &y[iby], iqs);

            if (sg_global_id == 0) {
                // cute::print(
                //     "mul_mat_vec_ocl: tid=%d item<2>=%d local_id<2>=%d row=%d, bpr=%d bps=%d eps=%d i=%d elem=%d "
                //     "ibx=%d iby=%d "
                //     "iqs=%d\n",  // partial_sum=%.8f\n",
                //     nd_item.get_global_linear_id(), nd_item.get_local_id(2), sg.get_local_linear_id(), row,
                //     blocks_per_row, blocks_per_subgroup, block_elements_per_subgroup, i, elem, ibx, iby,
                //     iqs /* , partial_sum */);
            }
        }
    }

    auto sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_cute_sycl_t vec_dot_cute_sycl /* , class Tensor_VX */
          /*, class TensorVY, class TensorDST */>
static void mul_mat_vec_cute(/* Tensor_VX tensor_vx, */ const void * vx, /* TensorVY */ const void * vy,
                             float * /* TensorDST */ dst, const int ncols, const int nrows,
                             const sycl::nd_item<3> & nd_item) {
    const int row = nd_item.get_group(2) * nd_item.get_local_range(1) + nd_item.get_local_id(1);

    if (nd_item.get_local_id(2) == 0) {
        cute::print("mul_mat_vec_cute: tid=%d item<2>=%d row=%d\n" /* partial_sum=%.8f\n" */,
                    nd_item.get_global_linear_id(), nd_item.get_local_id(2), row);
    }
    syncthreads();

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

    // using Element_VX        = typename Tensor_VX::value_type;
    // using Copy_Thread_Shape = cute::Shape<cute::Int<WARP_SIZE>, cute::_1>;
    // using Traits_Load_VX    = cute::Copy_Traits<cute::XE_2D_U8x1x32_LD_N, Tensor_VX>;
    // using Atom_Load_VX      = cute::Copy_Atom<Traits_Load_VX, Element_VX>;

    // auto tiled_copy_load_vx =
    //     make_tiled_copy(Atom_Load_VX{}.with(tensor_vx), cute::Layout<Copy_Thread_Shape>{},
    //                     cute::make_layout(cute::shape_div(typename Traits_Load_VX::BlockShape{}, Copy_Thread_Shape{})));

    // auto         subgroup               = nd_item.get_sub_group();
    // auto         first_thread_in_sg_idx = subgroup.get_group_linear_id() * WARP_SIZE;
    // cute::Tensor tCgA                   = thr_mma.partition_A(gA);

    // auto         thr_copy_A = tiled_copy_load_vx.get_slice(nd_item.get_local_id(2));
    // cute::Tensor tVX =
    //     cute::make_tensor<Element_VX>(cute::make_shape(cute::C<Atom_Load_VX::NumValDst>{}, cute::_1{}, cute::_1{}));

    // auto blk_load_s = cute::get_pvc_tensor(tensor_vx.shape());
    // auto thread_s   = thr_copy_A.partition_S(blk_load_s(cute::_, cute::_, 0));

    // clang-format off
#if 0
    if (cute::thread0()) {
        cute::print("tensor_vx: "); cute::print(tensor_vx); cute::print("\n");
        cute::print("thr_copy_A: "); cute::print(thr_copy_A); cute::print("\n");
        // cute::print("cute_tensor_vy: "); cute::print(cute_tensor_vy); cute::print("\n");
        // cute::print("cute_tensor_dst: "); cute::print(cute_tensor_dst); cute::print("\n");
        cute::print("thr_copy_A: "); cute::print(thr_copy_A); cute::print("\n");
        // cute::print("blk_load_s: "); cute::print(blk_load_s); cute::print("\n");
        // cute::print("thread_s: "); cute::print(thread_s); cute::print("\n");
        cute::print("\n\n");
    }
    syncthreads();
    if (cute::thread(1 * WARP_SIZE)) {
        cute::print("tensor_vx: "); cute::print(tensor_vx); cute::print("\n");
        cute::print("thr_copy_A: "); cute::print(thr_copy_A); cute::print("\n");
        // cute::print("cute_tensor_vy: "); cute::print(cute_tensor_vy); cute::print("\n");
        // cute::print("cute_tensor_dst: "); cute::print(cute_tensor_dst); cute::print("\n");
    }
#endif
    // clang-format on

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    // Prefetch 0, 1 from B two iby[values]
    // Prefetch 0, 1 from A two ibx[] values

    for (size_t i = nd_item.get_local_id(2) / block_elements_per_subgroup; i < blocks_per_row;
         i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        for (size_t elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            // x block quant index when casting the quants to int
            const int iqs = elem + vdr * (nd_item.get_local_id(2) % block_elements_per_subgroup);

            partial_sum += vec_dot_cute_sycl(&x[ibx], &y[iby], iqs);

            if (nd_item.get_local_id(2) == 0) {
                cute::print(
                    "mul_mat_vec_cute: tid=%d item<2>=%d row=%d, bpr=%d bps=%d eps=%d i=%d elem=%d ibx=%d iby=%d "
                    "iqs=%d,partial_sum=%.8f\n",
                    nd_item.get_global_linear_id(), nd_item.get_local_id(2), row, blocks_per_row, blocks_per_subgroup,
                    block_elements_per_subgroup, i, elem, ibx, iby, iqs, partial_sum);
            }
        }
    }

    auto sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum, std::plus<>());

    if (nd_item.get_local_id(2) == 0) {
        dst[row] = sum;
    }
}

static __dpct_inline__ float vec_dot_q4_0_q8_1_impl(const int * v, const int * u, const float & d4,
                                                    const sycl::half2 & ds8) {
    int sumi = 0;

#pragma unroll
    for (size_t i = 0; i < VDR_Q4_0_Q8_1_MMVCUTE; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

    const sycl::float2 ds8f = ds8.convert<float, sycl::rounding_mode::automatic>();

    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x() - (8 * VDR_Q4_0_Q8_1_MMVCUTE / QI4_0) * ds8f.y());
}

static __dpct_inline__ float vec_dot_q4_0_q8_1(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
                                               const int & iqs) {
    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq;
    int                v[VDR_Q4_0_Q8_1_MMVCUTE];

#ifdef __SYCL_DEVICE_ONLY__
    vector_t<int, 2> coord = { iqs, 0 };
    *reinterpret_cast<vector_t<ushort, 2> *>(v) =
        // u8 -> atomic value to bytes
        // m1 -> vertical dimension
        // k32 -> k-dimension size (also deifnes the stride of the values received by threads)
        // v2 -> 2 values
        // This translates to 64 bytes loaded, in two pairs of two bytes because:
        //   k32 * v2 = 64 u8 values,
        //   v2 = two blocks loaded 64 / 2 -> 32 u8 values are loaded per block,
        //   32 / WARP_SIZE = 2 values per thread
        detail::__builtin_IB_subgroup_block_read_flat_u8_m1k32v2((long) (vbq), 512 * sizeof(uint8_t) - 1, 16 - 1,
                                                                 512 * sizeof(uint8_t) - 1, coord);
    // baseoffset ALWAYS BEGINNING OF MEM REGION
    // Dimensions are defined by builtin (1k32v2) 1 x (32 * 2)
    // width (bytes)     -> actual width in bytes of the memory region
    // height (elements) -> 1
    // pitch (bytes)     -> helps defining the subblock of memory to access
    // x,y coord like system that helps identifying the block to load. Row major.
#endif

    int u[2 * VDR_Q4_0_Q8_1_MMVCUTE];  // q8 bytes == 2 * q4 bytes

#pragma unroll
    for (size_t i = 0; i < VDR_Q4_0_Q8_1_MMVCUTE; ++i) {
        v[i]         = get_int_from_uint8(bq4_0->qs, iqs + i);         // A
        u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);  // B
        u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }

    return vec_dot_q4_0_q8_1_impl(v, u, bq4_0->d, bq8_1->ds);
}

// NOTE: Will only work with GGML_SYCL_DISABLE_OPT=1 for now
static void mul_mat_vec_cute_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                            const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    // TODO: What is the purpose of GGML_SYCL_MMV_Y
    const int        block_num_y    = safe_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t cute_sg_per_wg = 16;
    GGML_ASSERT(block_num_y % cute_sg_per_wg == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
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
    printf("nrows=%d, ncols=%d\n", nrows, ncols);
    printf("global_size=%zu,%zu,%zu, local_size=%zu,%zu,%zu\n", global_size[0], global_size[1], global_size[2],
           wg_size[0], wg_size[1], wg_size[2]);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, wg_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             // mul_mat_vec_cute<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVCUTE, vec_dot_q4_0_q8_1>(
                             //     /* cute_tensor_vx, */ vx, vy, dst, ncols, nrows, nd_item);

                             mul_mat_vec_ocl<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVCUTE, vec_dot_q4_0_q8_1>(
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
