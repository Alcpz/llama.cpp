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

#include "dpct/helper.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml.h"
#include "quants.hpp"
#include "syclcompat/math.hpp"

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

namespace detail {
SYCL_EXTERNAL extern "C" int __builtin_IB_dp4a_ss(int c, int a, int b) __attribute__((const));
}

template <typename... T> void print(const char * format, const T &... t) {
    cute::print("\nIdx: (");
    cute::print(syclcompat::global_id::x());
    cute::print(",");
    cute::print(syclcompat::local_id::x());
    cute::print(") ");
    cute::print(format);
    cute::print(" ");
    ((cute::print(t), cute::print(" ")), ...);
}

template <typename... T> void print(const int i, const char * format, const T &... t) {
    if (syclcompat::local_id::x() == (size_t) i) {
        print(format, t...);
    }
}

template <int ElementSize, int Cols, int Rows, int Values> struct BlockLayout {
    static constexpr int element_size = ElementSize;
    static constexpr int cols         = Cols;
    static constexpr int rows         = Rows;
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
static __dpct_inline__ void get_quant_tile(const void * weights, size_t ncols, size_t nrows, coord_t coord, T * tile) {
#ifdef __SYCL_DEVICE_ONLY__
    using namespace cute::detail;

    // Width is expected in bytes. Quants are packed in bytes, 1 col == 1 nibble (q4_K)
    size_t width = ncols / (block_q_t::traits::qr);
    // XeSubgroup2DBlockLoad<Bytes,K,M,V>
    XeSubgroup2DBlockLoad<block_layout::element_size, block_layout::cols, block_layout::rows, block_layout::values>()(
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

template <typename T> static __dpct_inline__ void store_tile(const void * dst, size_t ncols, coord_t coord, T * tile) {
#ifdef __SYCL_DEVICE_ONLY__
    using namespace cute::detail;
    constexpr int nrows = 1;

    // Width is expected in bytes. Quants are packed in bytes, 1 col == 1 nibble (q4_K)
    size_t width = ncols * sizeof(float);
    // XeSubgroup2DBlockLoad<Bytes,K,M,V>
    XeSubgroup2DBlockStore<4, 16, 1, 1>()(dst, width, nrows, width, coord, tile);
#else
    (void) dst;
    (void) ncols;
    (void) coord;
    (void) tile;
    GGML_ABORT("Host code should not get here");
#endif
}

template <int tile_height> struct BlockLoadType {};

template <> struct BlockLoadType<1> {
    using T = ushort[1];
};

template <> struct BlockLoadType<2> {
    using T = ushort2;
};

template <> struct BlockLoadType<4> {
    using T = ushort4;
};

template <> struct BlockLoadType<8> {
    using T = ushort8;
};

template <> struct BlockLoadType<16> {
    using T = ushort16;
};

template <int tile_height> using block_load_t = typename BlockLoadType<tile_height>::T;

template <int tile_height> struct LayoutTraits {
    // Block Load
    static constexpr int bytes   = 2;
    static constexpr int rows    = tile_height;
    static constexpr int columns = 16;
    static constexpr int values  = 1;

    using QK_Layout = BlockLayout<bytes, columns, rows, values>;
    using Q8_Layout = BlockLayout<2 * bytes, columns, 1, values>;
    using QK_tile_t = block_load_t<tile_height>;

    // Tiled GemV traits
    static constexpr size_t coord_stride = columns;

    template <size_t prefetch_pipeline> static constexpr size_t prefetch_offset = coord_stride * prefetch_pipeline;

    template <typename block_q_t> __dpct_inline__ static size_t coord_range(size_t ncols) {
        return ncols / (QK_Layout::element_size * block_q_t::traits::qr);
    }
};

__dpct_inline__ static int32_t unpack_q4_tile(uint16_t q4_tile) {
    return ((q4_tile >> 12) & 0x0F) << 16 | ((q4_tile >> 8) & 0x0F) << 24 | ((q4_tile >> 4) & 0x0F) |
           (q4_tile & 0x0F) << 8;
}

// Hardcoded for the ease of development
// TODO: ncols < 64. Current intrinsic reads 16 int columns per load.
// Actual width of the data is ncols / qr -> for each block 512 / 2 = 256 bytes
// Current block load (m16, k16, v1) reads 16 rows, and from each row loads 16 ints -> 16 * 16 * 4 =
// 16 * 64 = 4096 bytes
// TODO: ensure rows * cols > 4096
template <typename Traits, size_t prefetch_pipeline>
__dpct_inline__ static void q4_K_q8_1_tiled_gemv(const void * weights, const void * input, float * dst,
                                                 const size_t ncols, const size_t nrows, const sycl::nd_item<1> & it,
                                                 std::integral_constant<reorder_kind_t, reorder_kind_t::LINEAR>) {
    using block_q_t  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>;
    using block_q8_t = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q8_1>;

    using bl_layout               = typename Traits::QK_Layout;  // bl = Block_load
    using q8_layout               = typename Traits::Q8_Layout;
    using q4_tile_t               = typename Traits::QK_tile_t;
    constexpr size_t coord_stride = Traits::coord_stride;
    size_t           coord_range  = Traits::template coord_range<block_q_t>(ncols);

    // Since local_range = WARP_RANGE
    const int     workgroup_id   = it.get_group_linear_id();
    auto          local_id       = it.get_local_linear_id();    // subgroup local id = workgroup local id
    constexpr int tile_height    = Traits::rows;
    const size_t  tile_row_begin = tile_height * workgroup_id;  // TODO: only supports a single sg per wg
    const int     blocks_per_row = ncols / block_q_t::traits::qk;

    const uint8_t *               weights_ptr = static_cast<const uint8_t *>(weights);
    sycl::vec<float, tile_height> partial_sums{ 0 };

    // INFO: Current blockloads grabs entire superblock every 4 iterations
    for (size_t tile_coord_begin = 0; tile_coord_begin < coord_range; tile_coord_begin += coord_stride) {
        const auto q4_coord = coord_t{ tile_coord_begin, tile_row_begin };
        const auto q8_coord = coord_t{ tile_coord_begin, 0 };

        q4_tile_t q4_tile;
        get_quant_tile<block_q_t, bl_layout>(weights, ncols, nrows, q4_coord, &q4_tile);

        int32_t q8_tile;
        get_quant_tile<block_q8_t, q8_layout>(input, ncols, 1, q8_coord, &q8_tile);

        const auto q4_qidx      = (tile_coord_begin + local_id) * Traits::bytes * block_q_t::traits::qr;
        const uint q8_dm_offset = block_q8_t::get_d_offset(1, ncols, q4_qidx / block_q8_t::traits::qk).first;

        const ggml_half2 * q8_dm =
            reinterpret_cast<const ggml_half2 *>(reinterpret_cast<const uint8_t *>(input) + q8_dm_offset);
        const float d8 = static_cast<float>(q8_dm->x());

        for (uint8_t i = 0; i < tile_height; i++) {
            const int          q4_block_idx = (tile_row_begin + i) * blocks_per_row + q4_qidx / block_q_t::traits::qk;
            const auto         scs_offsets  = block_q_t::get_d_offset(nrows, ncols, q4_block_idx);
            const uint8_t *    scales       = weights_ptr + scs_offsets.first;
            const ggml_half2 * dm4          = reinterpret_cast<const ggml_half2 *>(weights_ptr + scs_offsets.second);
            const sycl::float2 dm4f         = dm4->convert<float, sycl::rounding_mode::automatic>();

            // INFO: Scales are loaded a lot of times due to how they can't be properly reused
            const int chunk_idx = (q4_qidx / block_q8_t::traits::qk) % 8;
            const int j         = chunk_idx;

            uint8_t sc;
            uint8_t m;
            if (j < 4) {
                const uint16_t aux = *reinterpret_cast<const uint16_t *>(&scales[2 * j]) & 0x3f3f;
                sc                 = aux & 0xFF;
                m                  = (aux >> 8) & 0xFF;
            } else {
                const uint16_t hbits = *reinterpret_cast<const uint16_t *>(&scales[(j - 4) * 2]);
                sc                   = ((hbits & 0x00c0) >> 2) | ((scales[j + 4] >> 0) & 0x0f);
                m                    = (hbits & 0xc000) >> 10 | ((scales[j + 4] >> 4) & 0x0f);
            }

            const int32_t v    = unpack_q4_tile(q4_tile[i]);
            const int     dot1 = detail::__builtin_IB_dp4a_ss(0, v, q8_tile);
            const int     dot2 = detail::__builtin_IB_dp4a_ss(0, 0x01010101, q8_tile);

            // // if (cute::thread(0)) {
            // // if (cute::thread(0) || cute::thread(1) || cute::thread(2) || cute::thread(3) ||
            // //     cute::thread(4) || cute::thread(5) || cute::thread(6) || cute::thread(7)) {
            // if (it.get_group_linear_id() == 0) {
            //     for (size_t j = 0; j < 32; j++) {
            //         if (j == local_id && i == 1 && (q4_coord[0] >= 0)) {
            //             // print("==============");
            //             // print("q4_coord: ", q4_coord[0] + local_id, q4_coord[1]);
            //             // print("q8_coord: ", q8_coord[0] + local_id, q8_coord[1]);
            //             // print("q4_tile: ", q4_tile[i]);
            //             // {
            //             //     uint8_t q1 = (q4_tile[i] >> 12) & 0x0F;
            //             //     uint8_t q2 = (q4_tile[i] >> 8) & 0x0F;
            //             //     uint8_t q3 = (q4_tile[i] >> 4) & 0x0F;
            //             //     uint8_t q4 = q4_tile[i] & 0x0F;
            //             //     print("  q4tile[]:", q1, q2, q3, q4);
            //             // }
            //             // print("q4_qidx:", q4_qidx, block_q_t::traits::qk, q4_qidx / block_q_t::traits::qk);
            //             // print("q4_block_idx:", q4_block_idx);
            //             // print("q8_tile: ", q8_tile, static_cast<int>(q8_tile));
            //             // print("chunk_idx:", chunk_idx);
            //             // print("scs_offsets:", scs_offsets.first, scs_offsets.second);
            //             // print("scales (sc, m):", sc, m);
            //             // print("v, u:", v, static_cast<int>(q8_tile));
            //             // {
            //                 // print("  id:", q4_qidx,gq4_qidx + 1, q4_qidx + 2, q4_qidx + 3);
            //                 // uint8_t q1 = (v >> 24) & 0xFF;
            //                 // uint8_t q2 = (v >> 16) & 0xFF;
            //                 // uint8_t q3 = (v >> 8) & 0xFF;
            //                 // uint8_t q4 = (v >> 0) & 0xFF;
            //                 // print(" v[]:", q1, q2, q3, q4);
            //                 // q1 = (q8_tile >> 24) & 0xFF;
            //                 // q2 = (q8_tile >> 16) & 0xFF;
            //                 // q3 = (q8_tile >> 8) & 0xFF;
            //                 // q4 = q8_tile & 0xFF;
            //                 // print(" u[]:", q1, q2, q3, q4);
            //             // }
            //             // print("dm4:", (float) (dm4f.x()), (float) (dm4f.y()));
            //             // print("dm8:", (float) (q8_dm->x()));
            //             // print("dots:", dot1, dot2);
            //             // print("sums:", (float)(d8 * (dot1 * sc)), (float)(d8 * (dot2 * m)));
            //             // print("operands:", dm4f.x(), d8, dot1, sc, dm4f.y(), d8, dot2, m);
            //         }
            //     }
            // }

            partial_sums[i] += dm4f.x() * d8 * (dot1 * sc) - dm4f.y() * d8 * (dot2 * m);
        }
    }

    // const uint32_t* ptr = reinterpret_cast<const uint32_t*>(input);
    // const uint8_t* ptru8 = reinterpret_cast<const uint8_t*>(input);
    // const int8_t* wtr = reinterpret_cast<const int8_t*>(weights);
    // if (cute::thread(0)) {
    //     for (size_t i = 0; i < ncols; i+=2) {
    //         // print("(short idx) qs[", i / 2, "]:", *reinterpret_cast<const uint16_t*>(wtr + i));
    //     }
    //     for (size_t i = 0; i < ncols; i++) {
    //         // print("qs[", i, "]:", wtr[i] & 0x0F, (wtr[i] >> 4) & 0x0F);
    //     }
    //     for (size_t i = 0; i < ncols / 4; i++) {
    //         // print("q8[", i, "]:", ptr[i]);
    //     }
    //     for (size_t i = 0; i < ncols; i++) {
    //         // print("q8_u8[", i, "]:", ptru8[i]);
    //     }
    // }

    // for (size_t j = 0; j < WARP_SIZE; j++) {
    //     if (j == local_id) {
    //         print("partial_sums[0]:", partial_sums[0]);
    //     }
    // }

#pragma unroll
    for (size_t i = 0; i < tile_height; i++) {
        partial_sums[i] = sycl::reduce_over_group(it.get_sub_group(), partial_sums[i], std::plus<>());
    }

    // TODO: Ensure not storing out of bounds (tile_row_begin + i < nrows)
    // INFO: Block Loads seem to yield worse results
    // if constexpr (tile_height == 16) {
    //     const auto output_coord = coord_t{ tile_row_begin, 0 };
    //     const float output = partial_sums[local_id];
    //     store_tile(dst, nrows, output_coord, &output);
    // } else {
    if (it.get_sub_group().leader()) {
#pragma unroll
        for (size_t i = 0; i < tile_height; i++) {
            dst[tile_row_begin + i] = partial_sums[i];
        }
    }
    // }
}

static __dpct_inline__ void decode_chunk_scales(int chunk_idx, const uint8_t * scales, uint8_t * sc_m) {
    if (chunk_idx < 4) {
        *reinterpret_cast<uint16_t *>(sc_m) = *reinterpret_cast<const uint16_t *>(&scales[2 * chunk_idx]) & 0x3f3f;
    } else {
        const uint16_t hbits = *reinterpret_cast<const uint16_t *>(&scales[(chunk_idx - 4) * 2]);
        sc_m[0]              = ((hbits & 0x00c0) >> 2) | (scales[chunk_idx + 4] & 0x0f);
        sc_m[1]              = ((hbits & 0xc000) >> 10) | ((scales[chunk_idx + 4] >> 4) & 0x0f);
    }
}

static __dpct_inline__ void branchless_decode_chunk_scales(int chunk_idx, const uint8_t * scales, uint8_t * sc_m) {
    // chunk_idx < 4
    uint16_t val_a = *reinterpret_cast<const uint16_t*>(&scales[2 * chunk_idx]) & 0x3f3f;

    // chunk_idx >= 4
    uint16_t hbits = *reinterpret_cast<const uint16_t*>(&scales[(chunk_idx - 4) * 2]);
    uint8_t b0 = ((hbits & 0x00c0) >> 2) | (scales[chunk_idx + 4] & 0x0f);
    uint8_t b1 = ((hbits & 0xc000) >> 10) | ((scales[chunk_idx + 4] >> 4) & 0x0f);
    uint16_t val_b = static_cast<uint16_t>(b0) | (static_cast<uint16_t>(b1) << 8);

    uint32_t mask = -(chunk_idx < 4);
    *reinterpret_cast<uint16_t*>(sc_m) = (val_a & mask) | (val_b & ~mask);
}

static __dpct_inline__ void decode_superblock_scale(const uint8_t * weights_ptr, size_t offset, sycl::float2 * dm4f) {
    const ggml_half2 * dm4 = reinterpret_cast<const ggml_half2 *>(weights_ptr + offset);
    *dm4f                   = dm4->convert<float, sycl::rounding_mode::automatic>();
}

template <typename Traits, size_t prefetch_pipeline>
__dpct_inline__ static void q4_K_q8_1_tiled_gemv(
    const void * weights, const void * input, float * dst, const size_t ncols, const size_t nrows,
    const sycl::nd_item<1> & it, std::integral_constant<reorder_kind_t, reorder_kind_t::LINEAR_BLOCK_LOAD>) {
    using block_q_t  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>;
    using block_q8_t = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q8_1>;

    using bl_layout               = typename Traits::QK_Layout;  // bl = Block_load
    using q8_layout               = typename Traits::Q8_Layout;
    using q4_tile_t               = typename Traits::QK_tile_t;
    constexpr size_t coord_stride = Traits::coord_stride;
    constexpr int    tile_height        = Traits::rows;
    constexpr size_t sblock_coord_width = (block_q_t::traits::qk / block_q_t::traits::qr) / Traits::bytes;

    // Since local_range = WARP_RANGE
    size_t           coord_range  = Traits::template coord_range<block_q_t>(ncols);
    const int        workgroup_id       = it.get_group_linear_id();
    auto             local_id           = it.get_local_linear_id();    // subgroup local id = workgroup local id
    const size_t     tile_row_begin     = tile_height * workgroup_id;  // NOTE: only supports a single sg per wg
    const int        blocks_per_row     = ncols / block_q_t::traits::qk;

    const uint8_t *               weights_ptr = static_cast<const uint8_t *>(weights);
    sycl::vec<float, tile_height> partial_sums{ 0 };

    // INFO: Current blockloads grabs entire superblock every 4 iterations
    for (size_t tile_coord_begin = 0; tile_coord_begin < coord_range; tile_coord_begin += sblock_coord_width) {

        const int          chunk_idx    = local_id / 2;
        const int          superblock_idx = tile_coord_begin / 64;
        const int          q8_block_idx = superblock_idx * 8 + chunk_idx;
        const uint         q8_dm_offset = block_q8_t::get_d_offset(1, ncols, q8_block_idx).first;
        const ggml_half2 * q8_dm =
            reinterpret_cast<const ggml_half2 *>(reinterpret_cast<const uint8_t *>(input) + q8_dm_offset);
        const float d8 = static_cast<float>(q8_dm->x());

        int32_t      dot1[tile_height] = { 0 };
        int32_t      dot2[tile_height] = { 0 };
        uint8_t      sc_m[2 * tile_height];
        sycl::float2 dm4f[tile_height];

#pragma unroll(sblock_coord_width / Traits::columns)
        for (size_t w = 0; w < sblock_coord_width / Traits::columns; w++) {
            q4_tile_t q4_tile;
            uint q8_tile;

            const int w_coord = tile_coord_begin + coord_stride * w;
            const auto q4_coord = coord_t{ w_coord, tile_row_begin };
            get_quant_tile<block_q_t, bl_layout>(weights, ncols, nrows, q4_coord, &q4_tile);

            const auto q8_coord = coord_t{ w_coord, 0 };
            get_quant_tile<block_q8_t, q8_layout>(input, ncols, 1, q8_coord, &q8_tile);

#pragma unroll(tile_height)
            for (size_t i = 0; i < tile_height; i++) {
                const size_t    q4_k_block_idx = (tile_row_begin + i) * blocks_per_row + superblock_idx;
                const auto      scs_offsets    = block_q_t::get_d_offset(nrows, ncols, q4_k_block_idx);
                const uint8_t * scales         = weights_ptr + scs_offsets.first;

                if (!w) {
                    decode_superblock_scale(weights_ptr, scs_offsets.second, &dm4f[i]);
                    branchless_decode_chunk_scales(chunk_idx, scales, &sc_m[2*i]);
                }

                const int32_t q4_val = unpack_q4_tile(q4_tile[i]);
                const int32_t q8_val = static_cast<int32_t>(q8_tile);
                dot1[i]              = detail::__builtin_IB_dp4a_ss(dot1[i], q4_val, q8_val);
                dot2[i]              = detail::__builtin_IB_dp4a_ss(dot2[i], 0x01010101, q8_val);

                // // if (cute::thread(0) || cute::thread(1) || cute::thread(2) || cute::thread(3)) {
                // if (!it.get_group_linear_id()) {
                //     for (size_t j = 0; j < WARP_SIZE; j++) {
                //         if (j == local_id && i == 1 && w < 4) {
                //             // print("==============");
                //             // print("tile", tile_coord_begin, "w", w, "i", i, "-", sc[i], m[i]);
                //             // print("q4_coord: ", q4_coord[0] + local_id, q4_coord[1]);
                //             // print("q8_coord: ", q8_coord[0] + local_id, q8_coord[1]);
                //             // print("q4_tile: ", q4_tile[i]);
                //             // {
                //             //     uint8_t q1 = (q4_tile[i] >> 12) & 0x0F;
                //             //     uint8_t q2 = (q4_tile[i] >> 8) & 0x0F;
                //             //     uint8_t q3 = (q4_tile[i] >> 4) & 0x0F;
                //             //     uint8_t q4 = q4_tile[i] & 0x0F;
                //             //     print("  q4tile[]:", q1, q2, q3, q4);
                //             // }
                //             // print("chunk_idx:", chunk_idx);
                //             // print("q4_k_block_idx:", q4_k_block_idx);
                //             // print("q8_block_idx:", q8_block_idx);
                //             // print("q8_tile: ", q8_tile, static_cast<int>(q8_tile));
                //             // print("q8_offset: ", q8offset, local_id, block_q8_t::traits::qi, w);
                //             // print("scs_offsets:", scs_offsets.first, scs_offsets.second);
                //             // print("scales (sc, m):", sc[i], m[i]);
                //             // print("v, u:", q4_val, static_cast<int>(q8_tile));
                //             // {
                //                 // print("  id:", 4 * q8_coord[0], 4 * q8_coord[0] + 1, 4 * q8_coord[0] + 2, 4 * q8_coord[0] + 3);
                //             //     uint8_t q1 = (q4_val >> 24) & 0xFF;
                //             //     uint8_t q2 = (q4_val >> 16) & 0xFF;
                //             //     uint8_t q3 = (q4_val >> 8) & 0xFF;
                //             //     uint8_t q4 = (q4_val >> 0) & 0xFF;
                //             //     print(" v[]:", q1, q2, q3, q4);
                //             //     q1 = (q8_tile >> 24) & 0xFF;
                //             //     q2 = (q8_tile >> 16) & 0xFF;
                //             //     q3 = (q8_tile >> 8) & 0xFF;
                //             //     q4 = q8_tile & 0xFF;
                //             //     print(" u[]:", q1, q2, q3, q4);
                //             // }
                //             // print("dm4:", (float) (dm4f[i].x()), (float) (dm4f[i].y()));
                //             // print("dm8:", (float) (q8_dm->x()));
                //             // print("dots:", dot1[i], dot2[i], a, b);
                //             //
                //             // const int a = detail::__builtin_IB_dp4a_ss(0, q4_val, q8_val);
                //             // const int b = detail::__builtin_IB_dp4a_ss(0, 0x01010101, q8_val);
                //             // print("operands:", q4_val, q8_val, dm4f[i].x(), d8, a, sc[i], dm4f[i].y(), d8, b, m[i]);
                //         }
                //     }
                // }

            }
        }

#pragma unroll(tile_height)
        for (size_t i = 0; i < tile_height; i++) {
            partial_sums[i] += dm4f[i].x() * d8 * (dot1[i] * sc_m[2*i])
                - dm4f[i].y() * d8 * (dot2[i] * sc_m[2*i+1]);
        }
    }

    // const uint32_t * ptr = reinterpret_cast<const uint32_t *>(input);
    // const uint8_t* ptru8 = reinterpret_cast<const uint8_t*>(input);
    // const int8_t *   wtr = reinterpret_cast<const int8_t *>(weights);
    // if (cute::thread(0)) {
    //     for (size_t i = 0; i < ncols; i += 2) {
    //         // print("(short idx) qs[", i / 2, "]:", *reinterpret_cast<const uint16_t*>(wtr + i));
    //     }
    //     for (size_t i = 0; i < ncols; i++) {
    //         // print("qs[", i, "]:", wtr[i] & 0x0F, (wtr[i] >> 4) & 0x0F);
    //     }
    //     for (size_t i = 0; i < ncols / 4; i++) {
    //     //     print("q8[", i, "]:", ptr[i]);
    //     }
    //     for (size_t i = 0; i < ncols; i++) {
    //         //  print("q8_u8[", i, "]:", ptru8[i]);
    //     }
    // }

    // for (size_t j = 0; j < WARP_SIZE; j++) {
    //     if (j == local_id) {
    //         print("partial_sums[0]:", partial_sums[0]);
    //     }
    // }

#pragma unroll(tile_height)
    for (size_t i = 0; i < tile_height; i++) {
        partial_sums[i] = sycl::reduce_over_group(it.get_sub_group(), partial_sums[i], std::plus<>());
    }

    // TODO: Ensure not storing out of bounds (tile_row_begin + i < nrows)
    // INFO: Block Loads seem to yield worse results
    //     if (it.get_sub_group().leader()) {
    // #pragma unroll
    //         for (size_t i = 0; i < tile_height; i++) {
    //             dst[tile_row_begin + i] = partial_sums[i];
    //         }
    // }

    // INFO: Seems equivalent to the for loop using only the leader
    if (local_id < tile_height) {
        dst[tile_row_begin + local_id] = partial_sums[local_id];
    }
}

template <typename Traits, size_t prefetch_pipeline, reorder_kind_t Kind>
void launch_q4_K_q8_1_tiled_gemv(sycl::queue * stream, const void * vx, const void * vy, float * dst, size_t ncols,
                                 size_t nrows, sycl::nd_range<1> launch_range) {
    stream->submit([=](sycl::handler & cgh) {
        cgh.parallel_for(launch_range, [=](sycl::nd_item<1> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            q4_K_q8_1_tiled_gemv<Traits, prefetch_pipeline>(vx, vy, dst, ncols, nrows, nd_item,
                                                            std::integral_constant<reorder_kind_t, Kind>{});
        });
    });
}

void mul_mat_q4_K_q8_1_tiled_gemv(const void * vx, const void * vy, float * dst, const size_t ncols, const size_t nrows,
                                  dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    int tile_height = g_ggml_sycl_gemv_tile_height;
    GGML_ASSERT(nrows % tile_height == 0);

    constexpr size_t prefetch_pipeline = 0;

    const sycl::range<1> global_size(nrows * WARP_SIZE / tile_height);
    const sycl::range<1> wg_size(WARP_SIZE);
    // std::cout << "\nGlobal_range: " << global_size[0] << " Local_range: " << wg_size[0] << std::endl;

    if (g_ggml_sycl_gemv_reorder_format == (int) reorder_kind_t::LINEAR) {
        // std::cout << "LINEAR" << std::endl;
        if (tile_height == 16) {
            launch_q4_K_q8_1_tiled_gemv<LayoutTraits<16>, prefetch_pipeline, reorder_kind_t::LINEAR>(
                stream, vx, vy, dst, ncols, nrows, sycl::nd_range<1>(global_size, wg_size));
        } else if (tile_height == 8) {
            launch_q4_K_q8_1_tiled_gemv<LayoutTraits<8>, prefetch_pipeline, reorder_kind_t::LINEAR>(
                stream, vx, vy, dst, ncols, nrows, sycl::nd_range<1>(global_size, wg_size));
        } else if (tile_height == 4) {
            launch_q4_K_q8_1_tiled_gemv<LayoutTraits<4>, prefetch_pipeline, reorder_kind_t::LINEAR>(
                stream, vx, vy, dst, ncols, nrows, sycl::nd_range<1>(global_size, wg_size));
        } else if (tile_height == 2) {
            launch_q4_K_q8_1_tiled_gemv<LayoutTraits<2>, prefetch_pipeline, reorder_kind_t::LINEAR>(
                stream, vx, vy, dst, ncols, nrows, sycl::nd_range<1>(global_size, wg_size));
        } else if (tile_height == 1) {
            launch_q4_K_q8_1_tiled_gemv<LayoutTraits<1>, prefetch_pipeline, reorder_kind_t::LINEAR>(
                stream, vx, vy, dst, ncols, nrows, sycl::nd_range<1>(global_size, wg_size));
        } else {
            GGML_ABORT("unsupported tile height");
        }
    } else if (g_ggml_sycl_gemv_reorder_format == (int) reorder_kind_t::LINEAR_BLOCK_LOAD) {
        // std::cout << "LINEAR_BLOCK_LOAD" << std::endl;
        if (tile_height == 16) {
            launch_q4_K_q8_1_tiled_gemv<LayoutTraits<16>, prefetch_pipeline, reorder_kind_t::LINEAR_BLOCK_LOAD>(
                stream, vx, vy, dst, ncols, nrows, sycl::nd_range<1>(global_size, wg_size));
        } else if (tile_height == 8) {
            launch_q4_K_q8_1_tiled_gemv<LayoutTraits<8>, prefetch_pipeline, reorder_kind_t::LINEAR_BLOCK_LOAD>(
                stream, vx, vy, dst, ncols, nrows, sycl::nd_range<1>(global_size, wg_size));
        } else if (tile_height == 4) {
            launch_q4_K_q8_1_tiled_gemv<LayoutTraits<4>, prefetch_pipeline, reorder_kind_t::LINEAR_BLOCK_LOAD>(
                stream, vx, vy, dst, ncols, nrows, sycl::nd_range<1>(global_size, wg_size));
        } else if (tile_height == 2) {
            launch_q4_K_q8_1_tiled_gemv<LayoutTraits<2>, prefetch_pipeline, reorder_kind_t::LINEAR_BLOCK_LOAD>(
                stream, vx, vy, dst, ncols, nrows, sycl::nd_range<1>(global_size, wg_size));
        } else if (tile_height == 1) {
            launch_q4_K_q8_1_tiled_gemv<LayoutTraits<1>, prefetch_pipeline, reorder_kind_t::LINEAR_BLOCK_LOAD>(
                stream, vx, vy, dst, ncols, nrows, sycl::nd_range<1>(global_size, wg_size));
        } else {
            GGML_ABORT("unsupported tile height");
        }
    } else {
        GGML_ABORT("unsupported reorder_kind in q4_K");
        GGML_UNREACHABLE();
    }
}
