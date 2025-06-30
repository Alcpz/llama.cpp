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

SYCL_EXTERNAL extern "C" int __builtin_IB_dp4a_ss(int c, int a, int b) __attribute__((const));

// SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u32_m16k16v1(
//     intptr_t baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, coord_t coord,
//     CacheControl cache_control));

// namespace cute {
// namespace detail {
// template <> struct XeSubgroup2DBlockPrefetch<4, 16, 16, 1> {
//     CUTE_HOST_DEVICE void operator()(const void * srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
//                                      coord_t coordinate) {
//         __builtin_IB_subgroup_block_read_prefetch_u32_m16k16v1((intptr_t) srcBasePointer, memoryWidth - 1,
//                                                                memoryHeight - 1, memoryPitch - 1, coordinate,
//                                                                CacheControl::kL1C_L3C);
//     }
// };
// }  // namespace detail
// }  // namespace cute

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

template <typename block_q_t, typename block_layout, typename T>
static __dpct_inline__ void get_quant_tile_tr(const void * weights, size_t ncols, size_t nrows, coord_t coord,
                                              T * tile) {
#ifdef __SYCL_DEVICE_ONLY__
    using namespace cute::detail;

    // Width is expected in bytes. Quants are packed in bytes, 1 col == 1 nibble (q4_K)
    size_t width = ncols / (block_q_t::traits::qr);
    // XeSubgroup2DBlockLoad<Bytes,M,K,V>
    XeSubgroup2DBlockLoadTranspose<block_layout::element_size, block_layout::rows, block_layout::cols,
                                   block_layout::values>()(weights, width, nrows, width, coord, tile);
#else
    (void) weights;
    (void) nrows;
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

template <int tile_height>
using block_load_t = typename BlockLoadType<tile_height>::T;

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
                                                 const size_t ncols, const size_t nrows, const sycl::nd_item<1> & it) {
    using block_q_t  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>;
    using block_q8_t = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q8_1>;

    using bl_layout               = typename Traits::QK_Layout;  // bl = Block_load
    using q8_layout               = typename Traits::Q8_Layout;
    using q4_tile_t               = typename Traits::QK_tile_t;
    constexpr size_t coord_stride = Traits::coord_stride;
    size_t           coord_range  = Traits::template coord_range<block_q_t>(ncols);

    // Since local_range = WARP_RANGE
    const int     workgroup_id   = it.get_group_linear_id();
    auto          wi_id          = it.get_local_linear_id();    // subgroup local id = workgroup local id
    constexpr int tile_height    = Traits::rows;
    const size_t  tile_row_begin = tile_height * workgroup_id;  // TODO: only supports a single sg per wg
    const int     blocks_per_row = ncols / block_q_t::traits::qk;

    const uint8_t *               weights_ptr = static_cast<const uint8_t *>(weights);
    sycl::vec<float, tile_height> partial_sums{ 0 };

    // INFO: Current blockloads grabs entire superblock every 4 iterations
    for (size_t tile_coord_begin = 0; tile_coord_begin < coord_range; tile_coord_begin += coord_stride) {
        const auto q4_coord = coord_t{ tile_coord_begin, tile_row_begin };
        const auto q8_coord = coord_t{ tile_coord_begin, 0 };

        // TODO: Make it trait dependent
        q4_tile_t q4_tile;
        get_quant_tile<block_q_t, bl_layout>(weights, ncols, nrows, q4_coord, &q4_tile);

        uint q8_tile;
        get_quant_tile<block_q8_t, q8_layout>(input, ncols, nrows, q8_coord, &q8_tile);

        const auto q4_qidx = (tile_coord_begin + wi_id) * Traits::bytes * block_q_t::traits::qr;

        const uint q8_dm_offset = block_q8_t::get_d_offset(1, ncols, q4_qidx / block_q8_t::traits::qk).first;

        const ggml_half2 * q8_dm =
            reinterpret_cast<const ggml_half2 *>(reinterpret_cast<const uint8_t *>(input) + q8_dm_offset);

        // if (cute::thread(0) || cute::thread(7)) {
        //     for (size_t i = 0; i < WARP_SIZE; i++) {
        //         if (i == wi_id) {
        //             print("=========");
        //             print("q4_coord: ", q4_coord[0] + wi_id, q4_coord[1]);
        //             print("q8_coord: ", q8_coord[0] + wi_id, q8_coord[1]);
        //             print("q4_tile: ", q4_tile[0], q4_tile[1]);
        //             print("q8_tile: ", q8_tile);
        //             print("q4_qidx: ", q4_qidx);
        //             print("q8_dm_offsets: ", q8_dm_offset);
        //             print("q8_dm:", (float)(q8_dm->x()));
        //         }
        //     }
        // }

#pragma unroll
        for (uint8_t i = 0; i < tile_height; i++) {
            const int          q4_block_idx = (tile_row_begin + i) * blocks_per_row + q4_qidx / block_q_t::traits::qk;
            const auto         scs_offsets  = block_q_t::get_d_offset(nrows, ncols, q4_block_idx);
            const uint8_t *    scales       = weights_ptr + scs_offsets.first;
            const ggml_half2 * dm4          = reinterpret_cast<const ggml_half2 *>(weights_ptr + scs_offsets.second);
            const sycl::float2 dm4f         = dm4->convert<float, sycl::rounding_mode::automatic>();

            // INFO: Most likely the cause of the slow down, each scale + min is two global loads
            const int chunk_idx = (q4_qidx / block_q8_t::traits::qk) % 8;
            const int j         = chunk_idx;

            uint8_t sc;
            uint8_t m;
            if (j < 4) {
                const uint16_t aux = *reinterpret_cast<const uint16_t*>(&scales[2 * j]) & 0x3f3f;
                sc = aux & 0xFF;
                m  = (aux >> 8) & 0xFF;
            } else {
                const uint16_t hbits = *reinterpret_cast<const uint16_t*>(&scales[(j - 4) * 2]);
                sc = ((hbits & 0x00c0) >> 2) | ((scales[j + 4] >> 0) & 0x0f);
                m = (hbits & 0xc000) >> 10 | ((scales[j + 4] >> 4) & 0x0f);
            }

            // if (cute::thread(0) || cute::thread(8)) {
            //     for (size_t k = 0; k < WARP_SIZE; k++) {
            //         if (k == wi_id) {
            //             if (j < 4) {
            //                 print("j, scale(2):", j, sc, m, " - ", scales[2 * j] & 0x3f, scales[2 * j + 1] & 0x3f);
            //             } else {
            //                 print("j, scale(4):", j, sc, m, " - ", (scales[j + 4] >> 0) & 0x0f, (scales[j + 4] >> 4) & 0x0f, (scales[(j - 4) * 2 + 1] & 0xc0) >> 2, (scales[(j - 4) * 2] & 0xc0) >> 2);
            //             }
            //         }
            //     }
            // }

            // uint8_t aux[2];
            // if (j < 4) {
            //     aux[0] = scales[j + 0] & 0x3f;
            //     aux[1] = scales[j + 4] & 0x3f;
            // } else {
            //     aux[0] = ((scales[j + 4] >> 0) & 0x0f) | ((scales[j - 4] & 0xc0) >> 2);
            //     aux[1] = ((scales[j + 4] >> 4) & 0x0f) | ((scales[j - 0] & 0xc0) >> 2);
            // }
            // const uint8_t sc = aux[0];
            // const uint8_t m  = aux[1];

            // if (cute::thread(0) || cute::thread(8)) {
            //     for (size_t k = 0; k < WARP_SIZE; k++) {
            //         if (k == wi_id) {
            //             if (j < 4) {
            //                 print("j, scale(", j, "):", sc, m);
            //             } else {
            //                 print("j, scale(", j, "):", sc, m);
            //             }
            //         }
            //     }
            // }

            // TODO: Adjust dp4a
            const int32_t v    = unpack_q4_tile(q4_tile[i]);
            const int     dot1 = __builtin_IB_dp4a_ss(0, v, q8_tile);
            const int     dot2 = __builtin_IB_dp4a_ss(0, 0x01010101, q8_tile);

            const float d8 = static_cast<float>(q8_dm->x());

            // if (cute::thread(16) || cute::thread(17)) {
            //     for (size_t j = 0; j < 32; j++) {
            //         if (j == wi_id && i < 1 && (q4_coord[0] == 0)) {
            //             print("==============");
            //             print("q4_coord: ", q4_coord[0] + wi_id, q4_coord[1]);
            //             print("q8_coord: ", q8_coord[0] + wi_id, q8_coord[1]);
            //             print("q4_tile: ", q4_tile[i]);
            //             {
            //                 uint8_t q1 = (q4_tile[i] >> 12) & 0x0F;
            //                 uint8_t q2 = (q4_tile[i] >> 8)  & 0x0F;
            //                 uint8_t q3 = (q4_tile[i] >> 4)  & 0x0F;
            //                 uint8_t q4 =  q4_tile[i]        & 0x0F;
            //                 print("  q4tile[]:", q1, q2, q3, q4);
            //             }
            //             print("q4_qidx:", q4_qidx, block_q_t::traits::qk, q4_qidx / block_q_t::traits::qk);
            //             print("q4_block_idx:", q4_block_idx);
            //             print("q8_tile: ", q8_tile, static_cast<int>(q8_tile));
            //             print("chunk_idx:", chunk_idx);
            //             print("scs_offsets:", scs_offsets.first, scs_offsets.second);
            //             print("scales (sc, m):", sc[0], m[0]);
            //             print("v, u:", v, static_cast<int>(q8_tile));
            //             {
            //                 print("  id:", q4_qidx, q4_qidx +1, q4_qidx +2, q4_qidx +3);
            //                 uint8_t q1 = (v >> 24) & 0xFF;
            //                 uint8_t q2 = (v >> 16) & 0xFF;
            //                 uint8_t q3 = (v >> 8)  & 0xFF;
            //                 uint8_t q4 = (v >> 0)  & 0xFF;
            //                 print(" v[]:", q1, q2, q3, q4);
            //                 q1 = (q8_tile >> 24) & 0xFF;
            //                 q2 = (q8_tile >> 16) & 0xFF;
            //                 q3 = (q8_tile >> 8)  & 0xFF;
            //                 q4 =  q8_tile        & 0xFF;
            //                 print(" u[]:", q1, q2, q3, q4);
            //             }
            //             print("dm4:", (float)(dm4f.x()), (float)(dm4f.y()));
            //             print("dm8:", (float)(q8_dm->x()));
            //             print("dots:", dot1, dot2);
            //             print("sums:", (float)(d8 * (dot1 * sc[0])), (float)(d8 * (dot2 * m[0])));
            //         }
            //     }
            // }

            partial_sums[i] += dm4f.x() * d8 * (dot1 * sc) - dm4f.y() * d8 * (dot2 * m);
        }
    }

    // const uint32_t* ptr = reinterpret_cast<const uint32_t*>(input);
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
    // }

    // for (size_t j = 0; j < WARP_SIZE; j++) {
    //     if (j == wi_id) {
    //         print("partial_sums[0]:", partial_sums[0]);
    //     }
    // }

    #pragma unroll
        for (uint8_t i = 0; i < tile_height; i++) {
            partial_sums[i] = sycl::reduce_over_group(it.get_sub_group(), partial_sums[i], std::plus<>());
        }

// #pragma unroll
//     for (uint8_t i = 0; i < tile_height; i++) {
// #pragma unroll
//         for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
//             partial_sums[i] += dpct::permute_sub_group_by_xor(it.get_sub_group(), partial_sums[i], mask);
//         }
//     }

    // for (size_t j = 0; j < WARP_SIZE; j++) {
    //     if (j == wi_id) {
    //         print("partial_sums[0]:", partial_sums[0]);
    //     }
    // }

    // TODO: Ensure not storing out of bounds (tile_row_begin + i < nrows)
    if (it.get_sub_group().leader()) {
#pragma unroll
        for (size_t i = 0; i < tile_height; i++) {
            dst[tile_row_begin + i] = partial_sums[i];
            // if (workgroup_id == 0)
            // print("partial_sums[", tile_row_begin + i, "]:", partial_sums[i]);
        }
    }
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

    if (tile_height == 16) {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<1>(global_size, wg_size),
                             [=](sycl::nd_item<1> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 q4_K_q8_1_tiled_gemv<LayoutTraits<16>, prefetch_pipeline>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                             });
        });
    } else if (tile_height == 8) {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<1>(global_size, wg_size),
                             [=](sycl::nd_item<1> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 q4_K_q8_1_tiled_gemv<LayoutTraits<8>, prefetch_pipeline>(vx, vy, dst, ncols, nrows,
                                                                                          nd_item);
                             });
        });
    } else if (tile_height == 4) {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<1>(global_size, wg_size),
                             [=](sycl::nd_item<1> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 q4_K_q8_1_tiled_gemv<LayoutTraits<4>, prefetch_pipeline>(vx, vy, dst, ncols, nrows,
                                                                                          nd_item);
                             });
        });
    } else if (tile_height == 2) {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<1>(global_size, wg_size),
                             [=](sycl::nd_item<1> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 q4_K_q8_1_tiled_gemv<LayoutTraits<2>, prefetch_pipeline>(vx, vy, dst, ncols, nrows,
                                                                                          nd_item);
                             });
        });
    } else if (tile_height == 1) {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<1>(global_size, wg_size),
                             [=](sycl::nd_item<1> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 q4_K_q8_1_tiled_gemv<LayoutTraits<1>, prefetch_pipeline>(vx, vy, dst, ncols, nrows,
                                                                                          nd_item);
                             });
        });
    } else {
        GGML_ABORT("unsupported tile height");
    }
}
