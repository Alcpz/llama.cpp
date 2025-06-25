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
 *    Tiled gemv for llama.cpp's q4_K X q8_1 quantized types
 *    Expected GEMV Layout: nrows x ncols : ncols x 1;
 **************************************************************************/

#include <sycl/sycl.hpp>
#include <utility>

// #include "builtins.hpp"
#include "cacheopts.hpp"
#include "dpct/helper.hpp"
#include "ggml-sycl/common.hpp"
#include "quants.hpp"

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

#include <cute/arch/copy_xe_builtin.hpp>
#include <cute/util/print.hpp>

#pragma clang diagnostic pop

using namespace cute::intel;

SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u32_m16k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, coord_t coord,
    CacheControl cache_control));

namespace cute {
namespace detail {
template <> struct XeSubgroup2DBlockPrefetch<4, 16, 16, 1> {
    CUTE_HOST_DEVICE void operator()(const void * srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
                                     coord_t coordinate) {
        __builtin_IB_subgroup_block_read_prefetch_u32_m16k16v1((intptr_t) srcBasePointer, memoryWidth - 1,
                                                               memoryHeight - 1, memoryPitch - 1, coordinate,
                                                               CacheControl::kL1C_L3C);
    }
};
}  // namespace detail
}  // namespace cute


template <typename... T> void print(const char * format, const T &... t) {
    cute::print("Idx: ");
    cute::print(syclcompat::local_id::x());
    cute::print(" ");
    cute::print(format);
    cute::print(" ");
    ((cute::print(t), cute::print(" ")), ...);
    cute::print("\n");
}

template <int ElementSize, int Cols, int Rows, int Values> struct BlockLayout {
    static constexpr int element_size = ElementSize;
    static constexpr int rows         = Rows;
    static constexpr int cols         = Cols;
    static constexpr int values       = Values;
};

// INFO: u32 k16 loads a whole superblock in two loads
template <typename block_q_t, typename block_layout>
static __dpct_inline__ void prefetch_quant_tile(const void * weights, size_t ncols, size_t nrows, coord_t coord
                                                /*, LSC_LDCC cache_policy */) {
#ifdef __SYCL_DEVICE_ONLY__
    using namespace cute::detail;

    size_t width = ncols / (block_q_t::traits::qr);
    XeSubgroup2DBlockPrefetch<block_layout::element_size, block_layout::cols, block_layout::rows,
                              block_layout::values>()(weights, width, nrows, width, coord);
#else
    (void) weights;
    (void) nrows;
    (void) ncols;
    (void) coord;
    GGML_ABORT("Host code should not get here");
#endif
}

template <typename block_q_t, typename block_layout, typename T>
static __dpct_inline__ void get_quant_tile(const void * weights, size_t ncols, size_t nrows, coord_t coord, T* tile) {
#ifdef __SYCL_DEVICE_ONLY__
    using namespace cute::detail;

    // Width is expected in bytes. Quants are packed in bytes, 1 col == 1 nibble (q4_K)
    size_t width = ncols / (block_q_t::traits::qr);
    // XeSubgroup2DBlockLoad<Bytes, K,M,V>
    XeSubgroup2DBlockLoad<block_layout::element_size, block_layout::rows, block_layout::cols, block_layout::values>()(
        weights, width, nrows, width, coord, tile);
#else
    (void) weights;
    (void) nrows;
    (void) ncols;
    (void) coord;
    (void) tile;
    GGML_ABORT("Host code should not get here");
#endif
}

template <typename block_q_t, typename block_layout, typename T>
static __dpct_inline__ void get_quant_tile_tr(const void * weights, size_t ncols, size_t nrows, coord_t coord, T* tile) {
#ifdef __SYCL_DEVICE_ONLY__
    using namespace cute::detail;

    // Width is expected in bytes. Quants are packed in bytes, 1 col == 1 nibble (q4_K)
    size_t width = ncols / (block_q_t::traits::qr);
    // XeSubgroup2DBlockLoad<Bytes, M,K,V>
    XeSubgroup2DBlockLoadTranspose<block_layout::element_size, block_layout::cols, block_layout::rows, block_layout::values>()(
        weights, width, nrows, width, coord, tile);
#else
    (void) weights;
    (void) nrows;
    (void) ncols;
    (void) coord;
    (void) tile;
    GGML_ABORT("Host code should not get here");
#endif
}

// Pass V directly.
__dpct_inline__ float vec_dot_q4_K_q8_1_1(const uint32_t q4, const uint8_t * scs, const ggml_half2 * dm4,
                                          const uint2& q8, const ggml_half2 ** q8_1_dms, const int iqs) {
    const int        block_iqs = iqs % 32;
    const uint16_t * scales    = (const uint16_t *) scs;

    uint16_t  aux[2];
    const int j = (QR4_K * ((block_iqs / 2) / (QI8_1 / 2))) / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }

    const uint8_t * sc = (const uint8_t *) aux;
    const uint8_t * m  = sc + 2;

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll(QR4_K)
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (q4 >> (4 * i)) & 0x0f0f0f0f;

        const int dot1 = dpct::dp4a(v0i, static_cast<int>(q8[i]), 0);         // simd dot product
        const int dot2 = dpct::dp4a(0x01010101, static_cast<int>(q8[i]), 0);  // sum of u

        auto d8 = static_cast<float>(q8_1_dms[i]->x());
        sumf_d += d8 * (dot1 * sc[i]);
        sumf_m += d8 * (dot2 * m[i]);  // multiply constant part of q4_k with sum of q8_1 values

        // if (cute::thread(0)) {
        //     auto wi_id = syclcompat::local_id::x();
        //     for (size_t index = 0; index < WARP_SIZE; index++) {
        //         if (index == wi_id) {
        //             print("== vec_dot_loop v0i:  ", wi_id, i, v0i, u[2 * i], d8[i]);
        //             print("vec_dot_loop v1i:  ", wi_id, i, v1i, u[2 * i + 1], d8[i]);
        //             print("vec_dot_loop sums:  ", wi_id, i, sumf_d, sumf_m);
        //         }
        //     }
        // }

    }

    const sycl::float2 dm4f = dm4->convert<float, sycl::rounding_mode::automatic>();

    return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

__dpct_inline__ float vec_dot_q4_K_q8_1_2(const uint2& q4, const uint8_t * scs, const ggml_half2 * dm4,
                                          const uint4& q8, const ggml_half2 ** q8_1_dms, const int iqs) {
    // todo: find a more elegant way to pass the q8_1 block scales
    const uint16_t * scales    = (const uint16_t *) scs;

    uint16_t  aux[2];
    const int j = (iqs % 32) / (32 / 8);
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }

    const uint8_t * sc = (const uint8_t *) aux;
    const uint8_t * m  = sc + 2;

    if (cute::thread(0)) {
        auto wi_id = syclcompat::local_id::x();
        for (size_t index = 0; index < WARP_SIZE; index++) {
            if (index == wi_id) {
                print("--- vec_dot_loop(0):  ");
                print("  iqs: ", iqs);
                print("    j: ", j);
                print("    v[0]: ", static_cast<int>(q4[0]));
                print("    u[0]: ", static_cast<int>(q8[0]));
                print("    u[2]: ", static_cast<int>(q8[2]));
                print("--- vec_dot_input(1): ");
                print("  iqs: ", iqs + 1);
                print("    j: ", j);
                print("    v[1]: ", static_cast<int>(q4[1]));
                print("    u[1]: ", static_cast<int>(q8[1]));
                print("    u[3]: ", static_cast<int>(q8[3]));
                print("--- vec_dot_loop(sc): ", sc[0], sc[1]);
                print("--- vec_dot_loop(m): ", m[0], m[1]);
            }
        }
    }

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (q4[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (q4[1] >> (4 * i)) & 0x0F0F0F0F;
        const int u0i = static_cast<int>(q8[2 * i + 0]);
        const int u1i = static_cast<int>(q8[2 * i + 1]);

        const int dot1 = dpct::dp4a(v1i, u1i, dpct::dp4a(v0i, u0i, 0));  // SIMD dot product
        const int dot2 = dpct::dp4a(0x01010101, u1i, dpct::dp4a(0x01010101, u0i, 0));

        float d8 = static_cast<float>(q8_1_dms[i]->x());
        sumf_d += d8 * (dot1 * sc[i]);
        sumf_m += d8 * (dot2 * m[i]);

    //     if (cute::thread(0)) {
    //         auto wi_id = syclcompat::local_id::x();
    //         for (size_t index = 0; index < WARP_SIZE; index++) {
    //             if (index == wi_id) {
    //                 print("== vec_dot_loop v0i:  ", wi_id, i, v0i, u0i, d8);
    //                 // print("== vec_dot_loop dot1:  ", wi_id, i, dot1_1, dot1_2, dot1_3, dot1);
    //                 // print("== vec_dot_loop sums_1:  ", wi_id, i, sumf_d_1, sumf_m_1);
    //                 print("vec_dot_loop v1i:  ", wi_id, i, v1i, u1i, d8);
    //                 print("vec_dot_loop sc, m:  ", wi_id, i, sc[i], m[i]);
    //                 // print("vec_dot_loop dot2:  ", wi_id, i, dot2_1, dot2_2, dot2_3, dot2);
    //                 // print("vec_dot_loop sums_2:  ", wi_id, i, sumf_d_2, sumf_m_2);
    //                 // print("vec_dot_loop sums:  ", wi_id, i, sumf_d, sumf_m);
    //             }
    //         }
    //     }

    }

    const sycl::float2 dm4f = dm4->convert<float, sycl::rounding_mode::automatic>();

    // if (cute::thread(0)) {
    //     for (size_t index = 0; index < WARP_SIZE; index++) {
    //     auto wi_id = syclcompat::local_id::x();
    //     if (index == wi_id) {
    //         print("vec_dot_end:  ", iqs, dm4f.x(), dm4f.y());
    //     }
    //     }
    // }

    return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

template <int vec_dot_ratio>
struct LayoutTraits;

template <>
struct LayoutTraits<1> {
    // Block Load
    static constexpr int bytes   = 4;
    static constexpr int rows    = 16;
    static constexpr int columns = 16;
    static constexpr int values  = 1;

    using Layout = BlockLayout<bytes, columns, rows, values>;

    // Tiled GemV traits
    static constexpr size_t coord_stride    = columns;
    static constexpr size_t ncols_stride    = columns * bytes;

    template <size_t prefetch_pipeline>
    static constexpr size_t prefetch_offset = coord_stride * prefetch_pipeline;

    template <typename block_q_t>
    __dpct_inline__ static size_t coord_range(size_t ncols) {
        return ncols / (Layout::element_size * block_q_t::traits::qr);
    }
};

// Hardcoded for the ease of development
// TODO: ncols < 64. Current intrinsic reads 16 int columns per load.
// Actual width of the data is ncols / qr -> for each block 512 / 2 = 256 bytes
// Current block load (m16, k16, v1) reads 16 rows, and from each row loads 16 ints -> 16 * 16 * 4 =
// 16 * 64 = 4096 bytes
// TODO: ensure rows * cols > 4096
template <size_t prefetch_pipeline>
__dpct_inline__ static void q4_K_q8_1_tiled_gemv_1(const void * weights, const void * input, float * dst,
                                                   const size_t ncols, const size_t nrows, const sycl::nd_item<1> & it) {
    using block_q_t    = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>;
    using block_q8_1_t = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q8_1>;

    const uint32_t * q8_1_qs = reinterpret_cast<const uint32_t *>(input);

    using Traits = LayoutTraits<1>;
    using bl_layout = typename Traits::Layout;  // bl = Block_load
    constexpr size_t prefetch_offset = Traits::template prefetch_offset<prefetch_pipeline>;
    constexpr size_t coord_stride = Traits::coord_stride;
    constexpr size_t ncols_stride = Traits::ncols_stride;
    size_t coord_range = Traits::template coord_range<block_q_t>(ncols);

    // Since local_range = WARP_RANGE
    const int    workgroup_id   = it.get_group_linear_id();
    auto         wi_id          = it.get_local_linear_id();    // subgroup local id = workgroup local id
    constexpr int tile_height   = Traits::rows;
    const size_t tile_row_begin = tile_height * workgroup_id;  // TODO: only supports a single sg per wg
    const int    blocks_per_row = ncols / block_q_t::traits::qk;

    // TODO: quantization of q8_1 inside the kernel
    // sycl::vec<float, 4> input_fp32_vals;

    auto get_q8_1_idx = [=](int column, bool high_bits) {
        // Q4_K Is internally divided in 4 chunks of packed quants
        // Layout is: [ 32 0, 33 1, 32 2, ..., 63 31, 123 64, 124 65 ...] inside each superblocks block
        constexpr size_t qs_per_chunk      = 64;
        constexpr size_t columns_per_chunk = 32;

        size_t block_idx           = column / columns_per_chunk;                 // which chunk we are in
        size_t offset_within_block = column % columns_per_chunk;                 // byte offset within block

        return block_idx * qs_per_chunk + offset_within_block + high_bits * 32;  // 64 quants per chunk / qr
    };

    // Warmup Prefetch Tile A
    // TODO: What is the prefetch layout for a tranposed load
    // NOTE: Load LayoutTraits<1> (half superblock every I-th block loads?)
    if constexpr (prefetch_pipeline < 2) {  // Superblocks are guaranteed to require two loads
        for (size_t tile_coord_begin = 0; tile_coord_begin < prefetch_offset; tile_coord_begin += coord_stride) {
            auto prefetch_coord = coord_t{ tile_coord_begin, tile_row_begin };
            prefetch_quant_tile<block_q_t, bl_layout>(weights, ncols, nrows, prefetch_coord);
        }
    } else {
        for (size_t tile_coord_begin = 0; tile_coord_begin < prefetch_offset && tile_coord_begin < coord_range;
             tile_coord_begin += coord_stride) {
            auto prefetch_coord = coord_t{ tile_coord_begin, tile_row_begin };
            prefetch_quant_tile<block_q_t, bl_layout>(weights, ncols, nrows, prefetch_coord);
        }
    }

    const uint8_t *               weights_ptr = static_cast<const uint8_t *>(weights);
    sycl::vec<float, tile_height> partial_sums{ 0 };
    // TODO: should we read via L1 and not L3 ? This way we keep the L1 for the current inputs
    // INFO: Current blockloads grabs entire superblock every 2 iterations
    for (size_t tile_col_begin = 0; tile_col_begin < (ncols / block_q_t::traits::qr); tile_col_begin += ncols_stride) {
        auto tile_coord_begin = tile_col_begin / bl_layout::element_size;
        if (tile_coord_begin + prefetch_offset < coord_range) {
            const auto prefetch_coord = coord_t{ tile_coord_begin + prefetch_offset, tile_row_begin };
            // if (cute::thread(0)) {
            //     print("prefetch_coord[0, 1]: ", prefetch_coord[0], prefetch_coord[1]);
            // }
            prefetch_quant_tile<block_q_t, bl_layout>(weights, ncols, nrows, prefetch_coord);
        }

        const auto q4_coord = coord_t{ tile_coord_begin, tile_row_begin };

        // if (ggid == 0) {
        //     for (size_t i = 0; i < WARP_SIZE; i++) {
        //         if (i == wi_id) {
        //             print("ncols: ", ncols);
        //             print("nrows: ", nrows);
        //             print("coord: ", q4_coord[0], q4_coord[1]);
        //         }
        //     }
        // }

        uint16     q4_k_tile;
        get_quant_tile<block_q_t, bl_layout>(weights, ncols, nrows, q4_coord, &q4_k_tile);
        const auto q4_iqs = (tile_col_begin / bl_layout::element_size) + wi_id;

        const int     qs_stride = wi_id * bl_layout::element_size;
        const coord_t qs_idxs   = { get_q8_1_idx(tile_col_begin + qs_stride, false),
                                    get_q8_1_idx(tile_col_begin + qs_stride, true) };
        const uint2   q8_qs     = { q8_1_qs[qs_idxs[0] / 4], q8_1_qs[qs_idxs[1] / 4] };

        const uint2 q8_dm_offsets = {
            block_q8_1_t::get_d_offset(1, ncols, qs_idxs[0] / block_q8_1_t::traits::qk).first,
            block_q8_1_t::get_d_offset(1, ncols, qs_idxs[1] / block_q8_1_t::traits::qk).first
        };

        const ggml_half2 * q8_dms[2] = {
            reinterpret_cast<const ggml_half2 *>(reinterpret_cast<const uint8_t *>(input) + q8_dm_offsets[0]),
            reinterpret_cast<const ggml_half2 *>(reinterpret_cast<const uint8_t *>(input) + q8_dm_offsets[1])
        };

        // if (ggid == 0) {
        //     for (size_t i = 0; i < WARP_SIZE; i++) {
        //         if (i == wi_id) {
        //             print("=========");
        //             print("coord: ", q4_coord[0], q4_coord[1]);
        //             print("q4_k_tile: ", q4_k_tile[0]);
        //             print("qs_idxs: ", qs_idxs[0], qs_idxs[1]);
        //             print("q8_blk: ", qs_idxs[0] / block_q8_1_t::traits::qk, qs_idxs[1] / block_q8_1_t::traits::qk);
        //             print("q8_qs: ", q8_qs[0], q8_qs[1], qs_idxs[0] / block_q8_1_t::traits::qk, qs_idxs[1] / block_q8_1_t::traits::qk);
        //             print("q8_dm_offsets: ", q8_dm_offsets[0], q8_dm_offsets[1]);
        //             print("q8_dms:", (float)(q8_dms[0]->x()), (float)(q8_dms[1]->x()));
        //         }
        //     }
        // }

#pragma unroll(16)
        for (uint8_t i = 0; i < tile_height; i++) {
            const int q4_block_idx = (tile_row_begin + i) * blocks_per_row +
                                     (tile_col_begin * block_q_t::traits::qr + wi_id * 4) / block_q_t::traits::qk;
            const auto         scs_offsets = block_q_t::get_d_offset(nrows, ncols, q4_block_idx);
            const uint8_t *    scales      = weights_ptr + scs_offsets.first;
            const ggml_half2 * dm          = reinterpret_cast<const ggml_half2 *>(weights_ptr + scs_offsets.second);

            // if (cute::thread(0) || cute::thread(1)) {
            //     for (size_t j = 0; j < WARP_SIZE; j++) {
            //         if (j == wi_id && i < 1) {
            //             print("q4_k_tile:", (int)q4_k_tile[i], q4_k_tile[i]);
            //             print("block_calc:", tile_row_begin, i, blocks_per_row, tile_col_begin, wi_id, 4, block_q_t::traits::qk);
            //             print("q4_block_idx:", q4_block_idx);
            //             print("scs_offsets:", scs_offsets.first, scs_offsets.second);
            //             print("scales:", scales[0]);
            //             print("dm:", static_cast<float>(dm[0][0]), static_cast<float>(dm[0][1]));
            //         }
            //     }
            // }

            partial_sums[i] += vec_dot_q4_K_q8_1_1(q4_k_tile[i], scales, dm, q8_qs, q8_dms, q4_iqs);
        }
    }

    // const int32_t* ptr = reinterpret_cast<const int32_t*>(input);
    // const int32_t* wtr = reinterpret_cast<const int32_t*>(weights);
    // if (cute::thread(0)) {
    //     for (size_t i = 0; i < ncols / 4; i++) {
    //         if (i > (256 / 4))
    //         print("mem[", i, "]:", ptr[i]);
    //         else
    //           print("mem[", i, "]:", wtr[i], ptr[i]);
    //     }
    // }

    // for (size_t j = 0; j < WARP_SIZE; j++) {
    //     if (j == wi_id) {
    //         print("partial_sums[0]:", partial_sums[0]);
    //     }
    // }

#pragma unroll(16)
    for (uint8_t i = 0; i < tile_height; i++) {
        partial_sums[i] = sycl::reduce_over_group(it.get_sub_group(), partial_sums[i], std::plus<>());
    }

    // for (size_t j = 0; j < WARP_SIZE; j++) {
    //     if (j == wi_id) {
    //         print("partial_sums[0]:", partial_sums[0]);
    //     }
    // }

    // TODO: Ensure not storing out of bounds (tile_row_begin + i < nrows)
    if (it.get_sub_group().leader()) {
#pragma unroll(16)
        for (uint8_t i = 0; i < tile_height; i++) {
            dst[tile_row_begin + i] = partial_sums[i];
            // print("dst[i]:", tile_row_begin, partial_sums[i], dst[tile_row_begin + i]);
        }
    }
}


template <>
struct LayoutTraits<2> {
    // Block Load
    static constexpr int bytes   = 4;
    static constexpr int rows    = 16;
    static constexpr int columns = 2;
    static constexpr int values  = 1;

    using Layout = BlockLayout<bytes, columns, rows, values>;

    // Tiled GemV traits
    static constexpr size_t coord_stride = columns;
    static constexpr size_t ncols_stride = columns * bytes;

    template <size_t prefetch_pipeline>
    static constexpr size_t prefetch_offset = coord_stride * prefetch_pipeline;

    template <typename block_q_t>
    __dpct_inline__ static size_t coord_range(size_t ncols) {
        return ncols / (Layout::element_size * block_q_t::traits::qr);
    }
};



template <size_t prefetch_pipeline>
__dpct_inline__ static void q4_K_q8_1_tiled_gemv_2(const void * weights, const void * input, float * dst,
                                                   const size_t ncols, const size_t nrows, const sycl::nd_item<1> & it) {
    using block_q_t    = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>;
    using block_q8_1_t = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q8_1>;

    // int32_t* ptr_a = const_cast<int32_t *>(reinterpret_cast<const int32_t*>(input));
    // int32_t* wtr_b = const_cast<int32_t *>(reinterpret_cast<const int32_t*>(weights));
    // if (cute::thread(0)) {
    //     for (size_t i = 0; i < ncols / 4; i++) {
    //         if (i > (256 / 4)) {
    //             ptr_a[i] = i;
    //         } else {
    //             ptr_a[i] = i;
    //             wtr_b[i] = i;
    //         }
    //     }
    // }

    const uint32_t * q8_1_qs = reinterpret_cast<const uint32_t *>(input);

    using Traits = LayoutTraits<2>;
    using bl_layout = typename Traits::Layout;  // bl = Block_load
    // constexpr size_t prefetch_offset = Traits::template prefetch_offset<prefetch_pipeline>;
    // constexpr size_t coord_stride = Traits::coord_stride;
    constexpr size_t ncols_stride = Traits::ncols_stride;
    // size_t coord_range = Traits::template coord_range<block_q_t>(ncols);

    // Since local_range = WARP_RANGE
    const int    workgroup_id   = it.get_group_linear_id();
    auto         wi_id          = it.get_local_linear_id();    // subgroup local id = workgroup local id
    constexpr int tile_height = Traits::rows;
    const size_t tile_row_begin = tile_height * workgroup_id;  // TODO: only supports a single sg per wg
    const int    blocks_per_row = ncols / block_q_t::traits::qk;

    // TODO: quantization of q8_1 inside the kernel
    // sycl::vec<float, 4> input_fp32_vals;

    auto get_q4_k_iqs = [=](int iqs) {
        // Q4_K Is internally divided in 4 chunks of packed quants
        // Layout is: [ 32 0, 33 1, 34 2, ..., 63 31, 123 64, 124 65 ...] inside each superblocks block
        constexpr size_t iqs_per_chunk  = 16;
        constexpr size_t ints_per_chunk = 8;

        size_t block_idx           = iqs / ints_per_chunk;       // which chunk we are in
        size_t offset_within_block = iqs % ints_per_chunk;       // byte offset within block

        return block_idx * iqs_per_chunk + offset_within_block;  // 16 int-quants per chunk / qr
    };

    // // Warmup Prefetch Tile A
    // // TODO: Test prefetch of Q8_1
    // // TODO: Tune Prefetch couple of blocks in advance
    // if constexpr (prefetch_pipeline < 2) {  // Superblocks are guaranteed to require two loads
    //     for (size_t tile_coord_begin = 0; tile_coord_begin < prefetch_offset; tile_coord_begin += coord_stride) {
    //         auto prefetch_coord = coord_t{ tile_coord_begin, tile_row_begin };
    //         prefetch_quant_tile<block_q_t, bl_layout>(weights, ncols, nrows, prefetch_coord);
    //     }
    // } else {
    //     for (size_t tile_coord_begin = 0; tile_coord_begin < prefetch_offset && tile_coord_begin < coord_range;
    //          tile_coord_begin += coord_stride) {
    //         auto prefetch_coord = coord_t{ tile_coord_begin, tile_row_begin };
    //         prefetch_quant_tile<block_q_t, bl_layout>(weights, ncols, nrows, prefetch_coord);
    //     }
    // }

    const uint8_t * weights_ptr  = static_cast<const uint8_t *>(weights);
    float           partial_sums = 0.0f;
    // TODO: should we read via L1 and not L3 ? This way we keep the L1 for the current inputs
    // INFO: Current blockloads grabs entire superblock every 2 iterations
    for (size_t tile_col_begin = 0; tile_col_begin < (ncols / block_q_t::traits::qr); tile_col_begin += ncols_stride) {
        auto tile_coord_begin = tile_col_begin / bl_layout::element_size;

        // if (tile_coord_begin + prefetch_offset < coord_range) {
        //     const auto prefetch_coord = coord_t{ tile_coord_begin + prefetch_offset, tile_row_begin };
        //     // if (cute::thread(0)) {
        //     //     print("prefetch_coord[0, 1]: ", prefetch_coord[0], prefetch_coord[1]);
        //     // }
        //     prefetch_quant_tile<block_q_t, bl_layout>(weights, ncols, nrows, prefetch_coord);
        // }

        const auto q4_coord = coord_t{ tile_coord_begin, tile_row_begin };

        uint2     q4_k_tile;
        get_quant_tile_tr<block_q_t, bl_layout>(weights, ncols, nrows, q4_coord, &q4_k_tile);
        const auto  q4_iqs = get_q4_k_iqs(q4_coord[0]);
        const uint4 q8_qs  = { q8_1_qs[q4_iqs], q8_1_qs[q4_iqs + 1],
                               q8_1_qs[q4_iqs + block_q8_1_t::traits::qi], q8_1_qs[q4_iqs + block_q8_1_t::traits::qi + 1] };

        const uint2 q8_dm_offsets = {
            block_q8_1_t::get_d_offset(1, ncols, q4_iqs / block_q8_1_t::traits::qi).first,
            block_q8_1_t::get_d_offset(1, ncols, (q4_iqs / block_q8_1_t::traits::qi) + 1).first
        };

        const ggml_half2 * q8_dms[2] = {
            reinterpret_cast<const ggml_half2 *>(reinterpret_cast<const uint8_t *>(input) + q8_dm_offsets[0]),
            reinterpret_cast<const ggml_half2 *>(reinterpret_cast<const uint8_t *>(input) + q8_dm_offsets[1])
        };

        if (cute::thread(0)) {
            for (size_t i = 0; i < WARP_SIZE; i++) {
                if (i == wi_id && q4_coord[0] < 5) {
                    print("=========");
                    print("coord: ", q4_coord[0], q4_coord[1]);
                    print("q4_k_tile: ", (int)q4_k_tile[0], (int)q4_k_tile[1]);
                    print("q8_qs: ", (int)q8_qs[0], (int)q8_qs[1], (int)q8_qs[2], (int)q8_qs[3]);
                    print("q8_dm_offsets: ", q8_dm_offsets[0], q8_dm_offsets[1]);
                    print("q8_dms:", (float)(q8_dms[0]->x()), (float)(q8_dms[1]->x()));
                }
            }
        }

        const int          q4_block_idx = (q4_coord[1] + wi_id) * blocks_per_row + q4_coord[0];
        const auto         scs_offsets  = block_q_t::get_d_offset(nrows, ncols, q4_block_idx);
        const uint8_t *    scales       = weights_ptr + scs_offsets.first;
        const ggml_half2 * dm           = reinterpret_cast<const ggml_half2 *>(weights_ptr + scs_offsets.second);

        // if (cute::thread(0) /* || cute::thread(1) */) {
        //     for (size_t i = 0; i < Traits::columns; i++) {
        //     for (size_t j = 0; j < WARP_SIZE; j++) {
        //         if (j == wi_id && i < 1) {
        //             print("q4_k_tile:", (int)q4_k_tile[i], q4_k_tile[i]);
        //             print("block_calc:", q4_coord[1], wi_id, blocks_per_row, q4_coord[0], q4_block_idx);
        //             print("q4_block_idx:", q4_block_idx);
        //             print("scs_offsets:", scs_offsets.first, scs_offsets.second);
        //             print("scales:", scales[0]);
        //             print("dm:", static_cast<float>(dm[0][0]), static_cast<float>(dm[0][1]));
        //         }
        //     }
        //     }
        // }

        partial_sums += vec_dot_q4_K_q8_1_2(q4_k_tile, scales, dm, q8_qs, q8_dms, q4_iqs);
    }

    // const int32_t* ptr = reinterpret_cast<const int32_t*>(input);
    // const int32_t* wtr = reinterpret_cast<const int32_t*>(weights);
    // if (cute::thread(0)) {
    //     for (size_t i = 0; i < ncols / 4; i++) {
    //         if (i > (256 / 4))
    //         print("mem[", i, "]:", ptr[i]);
    //         else
    //           print("mem[", i, "]:", wtr[i], ptr[i]);
    //     }
    // }

    // for (size_t j = 0; j < WARP_SIZE; j++) {
    //     if (j == wi_id) {
    //         print("partial_sums[0]:", partial_sums[0]);
    //     }
    // }

// #pragma unroll(16)
//     for (uint8_t i = 0; i < tile_height; i++) {
//         partial_sums = sycl::reduce_over_group(it.get_sub_group(), partial_sums, std::plus<>());
//     }

    // for (size_t j = 0; j < WARP_SIZE; j++) {
    //     if (j == wi_id) {
    //         print("partial_sums[0]:", partial_sums[0]);
    //     }
    // }

    // TODO: Ensure not storing out of bounds (tile_row_begin + i < nrows)
    dst[tile_row_begin + wi_id] = partial_sums;

    for (size_t j = 0; j < WARP_SIZE; j++) {
        if (j == wi_id) {
            print("dst[]:", tile_row_begin + wi_id, partial_sums, dst[tile_row_begin + wi_id]);
        }
    }

}

void mul_mat_q4_K_q8_1_tiled_gemv(const void * vx, const void * vy, float * dst, const size_t ncols, const size_t nrows,
                                  dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr int tile_height = LayoutTraits<1>::rows;
    GGML_ASSERT(nrows % tile_height == 0);

    constexpr size_t prefetch_pipeline = 0;

    const sycl::range<1> global_size(nrows);
    const sycl::range<1> wg_size(tile_height);
    // std::cout << "\nGlobal_range: " << global_size[0] << " Local_range: " << wg_size[0] << std::endl;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(global_size, wg_size),
                         [=](sycl::nd_item<1> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             q4_K_q8_1_tiled_gemv_2<prefetch_pipeline>(vx, vy, dst, ncols, nrows,
                                                                                     nd_item);
                         });
    });
}
