/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  MIT License
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  gemv_q4_k_q8_1.cpp
 *
 *  Description:
 *    Tiled gemv for llama.cpp's q6_K X q8_1 quantized types
 *     Expected GEMV Layout: nrows x ncols : ncols x 1;
 **************************************************************************/

#include <sys/types.h>

#include <cstdint>
#include <sycl/aliases.hpp>
#include <sycl/sycl.hpp>

#include "builtins.hpp"
#include "cacheopts.hpp"
#include "dpct/helper.hpp"
#include "quants.hpp"

constexpr size_t tile_height = 16;

// INFO: u32 k16 loads a whole superblock in two loads
template <typename block_q_t>
static __attribute__((always_inline)) inline void prefetch_quant_tile(const void * weights, size_t ncols, size_t nrows,
                                                                      uint2 coord, LSC_LDCC cache_policy) {
#ifdef __SYCL_DEVICE_ONLY__
    size_t width = ncols / (block_q_t::traits::qr);
    __builtin_IB_subgroup_block_read_prefetch_u32_m16k16v1((intptr_t) weights, width - 1, nrows - 1, width - 1, coord,
                                                           cache_policy);
#else
    (void) weights;
    (void) nrows;
    (void) ncols;
    (void) coord;
    (void) cache_policy;
    GGML_ABORT("Host code should not get here");
#endif
}

template <typename block_q_t>
static __attribute__((always_inline)) inline uint16 get_quant_tile(const void * weights, size_t ncols, size_t nrows,
                                                                   uint2 coord) {
#ifdef __SYCL_DEVICE_ONLY__
    // Width is expected in bytes. Quants are packed in bytes, 1 col == 1 nibble (q4_K)
    size_t width = ncols / (block_q_t::traits::qr);
    auto   values =
        __builtin_IB_subgroup_block_read_flat_u32_m16k16v1((intptr_t) weights, width - 1, nrows - 1, width - 1, coord);
    return *reinterpret_cast<uint16 *>(&values);
#else
    (void) weights;
    (void) nrows;
    (void) ncols;
    (void) coord;
    GGML_ABORT("Host code should not get here");
#endif
}

// INFO: u16 k16 loads a two q8_1 blocks per load
static __attribute__((always_inline)) inline void prefetch_q8_tile(const void * weights, size_t ncols, size_t nrows,
                                                                   uint2 coord, LSC_LDCC cache_policy) {
#ifdef __SYCL_DEVICE_ONLY__
    __builtin_IB_subgroup_block_read_prefetch_u32_m1k16v1((intptr_t) weights, ncols - 1, nrows - 1, ncols - 1, coord,
                                                          cache_policy);
#else
    (void) weights;
    (void) nrows;
    (void) ncols;
    (void) coord;
    (void) cache_policy;
    GGML_ABORT("Host code should not get here");
#endif
}

static __attribute__((always_inline)) inline uint32_t get_q8_tile(const void * weights, size_t ncols, size_t nrows,
                                                                  uint2 coord) {
#ifdef __SYCL_DEVICE_ONLY__
    auto value =
        __builtin_IB_subgroup_block_read_flat_u32_m1k16v1((intptr_t) weights, ncols - 1, nrows - 1, ncols - 1, coord);
    return value;
#else
    (void) weights;
    (void) nrows;
    (void) ncols;
    (void) coord;
    GGML_ABORT("Host code should not get here");
#endif
}

// Pass V directly.
static inline float vec_dot_q4_K_q8_1(const void * __restrict__ vbq, const int ibx_offset, const int d_offset,
                                      const int8_t * q8_1_quant_ptr, const sycl::half2 * q8_1_ds, const int & iqs,
                                      int nblocks) {
    const int ib = ibx_offset / (QK_K / 2);

    const uint8_t *    base           = static_cast<const uint8_t *>(vbq);
    const uint8_t *    qs             = base + ibx_offset;
    const int          total_qs_bytes = nblocks * (QK_K / 2);
    const uint8_t *    scs            = base + total_qs_bytes + ib * K_SCALE_SIZE;
    const ggml_half2 * dm             = reinterpret_cast<const ggml_half2 *>(base + d_offset);

    const int        bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int *      q4         = (const int *) (qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const uint16_t * scales     = (const uint16_t *) scs;

    int   v[2];
    int   u[2 * QR4_K];
    float d8[QR4_K];

    v[0] = q4[0];
    v[1] = q4[4];

    uint16_t  aux[2];
    const int j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
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
        const int8_t * quant_base_ptr = q8_1_quant_ptr + (bq8_offset + i) * QK8_1;
        sycl::half2    ds_values      = *(q8_1_ds + bq8_offset + i);

        d8[i] = ds_values[0];

        const int * q8 = (const int *) quant_base_ptr + ((iqs / 2) % 4);
        u[2 * i + 0]   = q8[0];
        u[2 * i + 1]   = q8[4];
    }

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    int       i   = 0;  // TOOD: Get rid of this as we are switching to a single value per vecdot
    const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
    const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;

    const int dot1 = dpct::dp4a(v1i, u[2 * i + 1], dpct::dp4a(v0i, u[2 * i + 0], 0));
    const int dot2 = dpct::dp4a(0x01010101, u[2 * i + 1], dpct::dp4a(0x01010101, u[2 * i + 0], 0));  // sum of u

    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values

    const sycl::float2 dm4f = dm->convert<float, sycl::rounding_mode::automatic>();

    return dm4f[0] * sumf_d - dm4f[1] * sumf_m;
}

// Hardcoded for the ease of development
// TODO: ncols < 64. Current intrinsic reads 16 int columns per load.
// Actual width of the data is ncols / qr -> for each block 512 / 2 = 256 bytes
// Current block load (m16, k16, v1) reads 16 rows, and from each row loads 16 ints -> 16 * 16 * 4 =
// 16 * 64 = 4096 bytes TODO: ensure rows * cols > 4096
template <size_t tile_height, size_t prefetch_pipeline>
__attribute__((always_inline)) inline static void q4_K_q8_1_tiled_gemv(const void * weights, const void * input,
                                                                       float * dst, const size_t ncols,
                                                                       const size_t             nrows,
                                                                       const sycl::nd_item<1> & it) {
    using block_q_t                       = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>;
    // intel_sub_group_i8_i4_matrix_mad_k32(short a, int 4 b, int c);
    // 1 x 32 : 32 x 16;
    // 1 x 32 is very doable, we are doing it even today
    // 32 x 16 is the problem
    // k16 * v1 * u32 = stride 16 * 4
    constexpr size_t ncols_stride         = 64;  //threads.u32 == 2048 bytes per block load
    constexpr size_t block_load_cell_size = 4;
    constexpr size_t coord_stride         = ncols_stride / block_load_cell_size;
    constexpr size_t prefetch_offset      = coord_stride * prefetch_pipeline;

    // Since local_range = WARP_RANGE
    auto         subgroup_id    = it.get_local_id(0);
    auto         wi_id          = it.get_local_linear_id();  // subgroup local id = workgroup local id
    const size_t tile_row_begin = tile_height * subgroup_id;
    const int    blocks_per_row = ncols / block_q_t::traits::qk;

    // TODO: quantization of q8_1 inside the kernel
    // sycl::vec<float, 4> input_fp32_vals;

    auto get_qs_index = [=](int column, bool high_bits) {
        constexpr size_t quants_per_block  = 64;                        // 32 low bits + 32 high bits per block
        constexpr size_t columns_per_block = 8;                         // 32 quants / 4 quants per u32 load = 8 columns

        size_t block_idx           = column / columns_per_block;        // which 64-quant block we are in
        size_t offset_within_block = (column % columns_per_block) * 4;  // byte offset within block

        //  Layout is: [ 32 0, 33 1, 32 2, ..., 63 31, 123 64, 124 65 ...] inside each superblocks block
        return block_idx * quants_per_block + offset_within_block + (high_bits ? 32 : 0);
    };

    // Warmup Prefetch Tile A
    // TODO: Tune Prefetch couple of blocks in advance
    // TODO: Start the prefetch of the next B Tile as well.
    for (size_t tile_col_begin = 0; tile_col_begin < prefetch_offset; tile_col_begin += coord_stride) {
        auto q4_prefetch_coord = uint2{ tile_col_begin, tile_row_begin };
        prefetch_quant_tile<block_q_t>(weights, ncols, nrows, q4_prefetch_coord, LSC_LDCC_L1C_L3C);
        auto q8_prefetch_coord = uint2{ get_qs_index(tile_col_begin, false), 0 };
        prefetch_quant_tile<block_q_t>(weights, ncols, 1, q8_prefetch_coord, LSC_LDCC_L1C_L3C);
        q8_prefetch_coord = uint2{ get_qs_index(tile_col_begin, true), 0 };
        prefetch_quant_tile<block_q_t>(weights, ncols, 1, q8_prefetch_coord, LSC_LDCC_L1C_L3C);
    }

    sycl::vec<float, tile_height> partial_sums{ 0 };
    // TODO: should we read via L1 and not L3 ? This way we keep the L1 for the current inputs
    // INFO: Current blockloads grabs entire superblock every 2 iterations (reorder chances)
    for (size_t tile_col_begin = 0; tile_col_begin < ncols; tile_col_begin += coord_stride) {
        // and L3 for the outputs to enable cache hits for the next layer.
        auto prefetch_coord = uint2{ tile_col_begin + prefetch_offset, tile_row_begin };
        prefetch_quant_tile<block_q_t>(weights, ncols, nrows, prefetch_coord, LSC_LDCC_L1C_L3C);
        // TODO: prefetch next q8

        auto q4_coord  = uint2{ tile_col_begin, tile_row_begin };
        auto q4_k_tile = get_quant_tile<block_q_t>(weights, ncols, nrows, q4_coord);
        // auto scales    = block_q_t::get_d_offset(nrows, ncols, block_index);
        // auto q4m       = ;
        // auto scs       = ;

        uint2 q8_qs = {
            get_q8_tile(weights, ncols, nrows, uint2{ get_qs_index(tile_col_begin, false), 0 }),
            get_q8_tile(weights, ncols, nrows, uint2{ get_qs_index(tile_col_begin, true), 0 })
        };

        // TODO: Load scales (Block load may not be feasible as scales are per 8 quants
        uint2 q8_dms = {};

        // TODO: If I am in the last K tile, prefetch my Q4_K scales
        //
#pragma unroll(16)
        for (uint8_t i = 0; i < tile_height; i++) {
            // TODO: check, are we sure that we can simply zero-extend 4 bit integers to 8 bit integers ?
            // whatever happened to two's complement ? I suppose that's how 4 bit negative integeters
            // are represented as well.

            // partial_sums[i] = vec_dot_q4_K_q8_1(q4_k_tile[i], dm, scs, &q8_qs, &q8_dms);
        }
    }

    // INFO: Seems that a simple store could be faster
    // #pragma unroll(tile_height)
    //     for (uint8_t i = 0; i <
    //         // select element id = sg.local_linear_
    //         // TODO: Why am i doing this ? this is li
    //         // the dpas instruction, might as well dpas directly ?
    //     }
    //
    // store 16 fp32 values. 1 value per thread
    //__builtin_IB_subgroup_block_write_flat_u32_m1k16v1(...)

    // TODO: Ensure not storing out of bounds (tile_row_begin + i < nrows)
    if (it.get_sub_group().leader()) {
#pragma unroll(16)
        for (uint8_t i = 0; i < tile_height; i++) {
            dst[tile_row_begin + i] = partial_sums[i];
        }
    }
}

void mul_mat_q4_K_q8_1_tiled_gemv(const void * vx, const void * vy, float * dst, const size_t ncols, const size_t nrows,
                                  dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    GGML_ASSERT(nrows % tile_height == 0);

    constexpr size_t tile_height       = 16;
    constexpr size_t prefetch_pipeline = 2;

    const sycl::range<1> global_size(nrows);
    const sycl::range<1> wg_size(tile_height);
    std::cout << "Global_range: " << global_size[0] << " Local_range: " << wg_size[0] << std::endl;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(global_size, wg_size),
                         [=](sycl::nd_item<1> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             q4_K_q8_1_tiled_gemv<tile_height, prefetch_pipeline>(vx, vy, dst, ncols, nrows, nd_item);
                         });
    });
}
