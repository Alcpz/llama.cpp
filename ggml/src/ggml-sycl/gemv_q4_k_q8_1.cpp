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

#include "builtins.hpp"
#include "cacheopts.hpp"
#include "dpct/helper.hpp"
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

#include <cute/util/debug.hpp>

#pragma clang diagnostic pop


constexpr size_t tile_height = 16;

// INFO: u32 k16 loads a whole superblock in two loads
template <typename block_q_t>
static __dpct_inline__ void prefetch_quant_tile(const void * weights, size_t ncols, size_t nrows, uint2 coord,
                                                LSC_LDCC cache_policy) {
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
static __dpct_inline__ uint16 get_quant_tile(const void * weights, size_t ncols, size_t nrows, uint2 coord) {
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


// Pass V directly.
static __dpct_inline__ float vec_dot_q4_K_q8_1(
    const uint32_t q4,
    const uint8_t* scales,
    const ggml_half2 * dm4,
    const uint2& q8,
    const ggml_half2 ** q8_1_dms,
    const int & iqs
) {

    int   v[1]  = { q4 };
    int   u[QR4_K]  = { q8[0], q8[1] };
    // TODO: Find a more elegant way to pass the q8_1 block scales
    float d8[QR4_K] = { q8_1_dms[0][0][0], q8_1_dms[1][0][0] };

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

    //return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, dm, d8);
    //
    //const int *__restrict__ v, const int *__restrict__ u,
    //const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
    // const sycl::half2 &dm4, const float *__restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll(QR4_K)
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = dpct::dp4a(v0i, u[i], 0);        // SIMD dot product
        const int dot2 = dpct::dp4a(0x01010101, u[i], 0); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const sycl::float2 dm4f =
        dm4->convert<float, sycl::rounding_mode::automatic>();

    return dm4f[0] * sumf_d - dm4f[1] * sumf_m;
}

// Hardcoded for the ease of development
// TODO: ncols < 64. Current intrinsic reads 16 int columns per load.
// Actual width of the data is ncols / qr -> for each block 512 / 2 = 256 bytes
// Current block load (m16, k16, v1) reads 16 rows, and from each row loads 16 ints -> 16 * 16 * 4 =
// 16 * 64 = 4096 bytes
// TODO: ensure rows * cols > 4096
template <size_t tile_height, size_t prefetch_pipeline>
__dpct_inline__ inline static void q4_K_q8_1_tiled_gemv(const void * weights, const void * input, float * dst,
                                                        const size_t ncols, const size_t nrows,
                                                        const sycl::nd_item<1> & it) {
    using block_q_t    = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>;
    using block_q8_1_t = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q8_1>;

    const uint8_t* q8_1_qs = reinterpret_cast<const uint8_t *>(input);

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

    if (cute::thread0()) {
      sycl::ext::oneapi::experimental::printf("\nHERE I AM");
    }

    auto get_q8_1_idx = [=](int column, bool high_bits) {
        // Q4_K Is internally divided in 4 chunks of packed quants
        // Layout is: [ 32 0, 33 1, 32 2, ..., 63 31, 123 64, 124 65 ...] inside each superblocks block
        constexpr size_t columns_per_chunk = 32;

        size_t block_idx           = column / columns_per_chunk;  // which chunk we are in
        size_t offset_within_block = column % columns_per_chunk;  // byte offset within block

        return block_idx * columns_per_chunk + offset_within_block + high_bits * 32; // 64 quants per chunk / qr
    };

    // Warmup Prefetch Tile A
    // TODO: Tune Prefetch couple of blocks in advance
    // TODO: Start the prefetch of the next B Tile as well.
    for (size_t tile_col_begin = 0; tile_col_begin < prefetch_offset; tile_col_begin += coord_stride) {
        auto q4_prefetch_coord = uint2{ tile_col_begin, tile_row_begin };
        prefetch_quant_tile<block_q_t>(weights, ncols, nrows, q4_prefetch_coord, LSC_LDCC_L1C_L3C);
    }

    const uint8_t * weights_ptr = static_cast<const uint8_t *>(weights);
    sycl::vec<float, tile_height> partial_sums{ 0 };
    // TODO: should we read via L1 and not L3 ? This way we keep the L1 for the current inputs
    // INFO: Current blockloads grabs entire superblock every 2 iterations (reorder chances)
    for (size_t tile_col_begin = 0; tile_col_begin < ncols; tile_col_begin += coord_stride) {
        // and L3 for the outputs to enable cache hits for the next layer.
        const auto prefetch_coord = uint2{ tile_col_begin + prefetch_offset, tile_row_begin };
        prefetch_quant_tile<block_q_t>(weights, ncols, nrows, prefetch_coord, LSC_LDCC_L1C_L3C);

        const auto q4_coord  = uint2{ tile_col_begin, tile_row_begin };
        const auto q4_k_tile = get_quant_tile<block_q_t>(weights, ncols, nrows, q4_coord);

        const uint2 qs_idxs = { get_q8_1_idx(tile_col_begin, false), get_q8_1_idx(tile_col_begin, true) };
        const uint2 q8_qs = { q8_1_qs[qs_idxs[0]], q8_1_qs[qs_idxs[1]] };

        const uint2 q8_dm_offsets = {
            block_q8_1_t::get_d_offset(nrows, ncols, q8_qs[0] / block_q8_1_t::traits::qk).first,
            block_q8_1_t::get_d_offset(nrows, ncols, q8_qs[1] / block_q8_1_t::traits::qk).first
        };
        const ggml_half2* q8_dms[2] = {
            reinterpret_cast<const ggml_half2*>(reinterpret_cast<const uint8_t*>(input) + q8_dm_offsets[0]),
            reinterpret_cast<const ggml_half2*>(reinterpret_cast<const uint8_t*>(input) + q8_dm_offsets[1])
        };

#pragma unroll(16)
        for (uint8_t i = 0; i < tile_height; i++) {
            const int q4_block_idx    = (tile_row_begin + i) * blocks_per_row + (tile_col_begin + wi_id * 4) / block_q_t::traits::qk;
            const auto scs_offsets = block_q_t::get_d_offset(nrows, ncols, q4_block_idx);
            const uint8_t* scales  = weights_ptr + scs_offsets.first;
            const ggml_half2 * dm  = reinterpret_cast<const ggml_half2 *>(weights_ptr + scs_offsets.second);

            partial_sums[i] += vec_dot_q4_K_q8_1(q4_k_tile[i], scales, dm, q8_qs, q8_dms, qs_idxs[0]);
        }
    }

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
