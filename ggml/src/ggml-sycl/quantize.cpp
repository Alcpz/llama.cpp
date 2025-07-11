#include "quantize.hpp"
#include "ggml-sycl/common.hpp"

template <int QUANT_BLOCK_TILE>
void quantize_q8_1(const float * __restrict__ x, void * __restrict__ vy, const int kx, const int kx_padded,
                   const sycl::nd_item<3> & item_ct1) {
    const int ix = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * QUANT_BLOCK_TILE;

    if (ix >= kx_padded) {
        return;
    }

    const int iy = item_ct1.get_local_range(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1);

    const int i_padded = iy * kx_padded + ix;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int                                   ib  = i_padded / QK8_1;  // block index
    const int                                   iqs = i_padded % QK8_1;  // quant index
    typedef sycl::vec<float, QUANT_BLOCK_TILE>  TC;
    typedef sycl::vec<int8_t, QUANT_BLOCK_TILE> TQ;
    TC                                          zeros;
    TQ                                          qzeros;
#pragma unroll
    for (int i = 0; i < QUANT_BLOCK_TILE; i++) {
        zeros[i]  = 0.f;
        qzeros[i] = 0;
    }
    const TC xi   = ix < kx ? *(const TC *) &x[iy * kx + ix] : zeros;
    float    sum  = xi[0];
    float    amax = sycl::fabs(xi[0]);
#pragma unroll
    for (int i = 1; i < QUANT_BLOCK_TILE; i++) {
        sum += xi[i];
        amax = sycl::fmax(sycl::fabs(xi[i]), amax);
    }
    sum  = warp_reduce_sum(sum, item_ct1);
    amax = warp_reduce_max(amax, item_ct1);

    const float d = amax / 127;
    TQ          q = qzeros;
    if (amax != 0.0f) {
#pragma unroll
        for (int i = 0; i < QUANT_BLOCK_TILE; i++) {
            q[i] = sycl::round(xi[i] / d);
        }
    }

    *(TQ *) &y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    reinterpret_cast<sycl::half &>(y[ib].ds.x()) = d;
    reinterpret_cast<sycl::half &>(y[ib].ds.y()) = sum;
}


// template <int ElementsPerWI>
// __dpct_inline__ void quantize_and_reorder_q8_1_linear(const float * __restrict__ x, void * reordered_q8_tensor,
//                                                       const int kx, const int kx_padded, const sycl::nd_item<1> & it) {
//     auto subgroup_id = it.get_group(0);
//     auto wi_id       = it.get_local_id(0);
//
//     const int num_blocks_per_row = kx / QK8_1;
//     auto      row                = subgroup_id / num_blocks_per_row;
//     auto      col                = subgroup_id % num_blocks_per_row;
//
//     auto row_offset = row * (kx_padded / QK8_1) * sizeof(block_q8_1);
//
//     auto ds_ptr = (sycl::half2 *) ((char *) reordered_q8_tensor + row_offset + kx + col * sizeof(sycl::half2));
//
//     sycl::vec<float, ElementsPerWI>  wi_f32_vals;
//     sycl::vec<int8_t, ElementsPerWI> quantized_values;
//
//     auto float_ptr_offset = subgroup_id * QK8_1 + ElementsPerWI * wi_id;
//     wi_f32_vals           = *reinterpret_cast<const sycl::vec<float, ElementsPerWI> *>(x + float_ptr_offset);
//
//     float sum  = 0.0f;
//     float amax = 0.0f;
//
// #pragma unroll(ElementsPerWI)
//     for (int i = 0; i < ElementsPerWI; i++) {
//         sum += wi_f32_vals[i];
//         amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
//         quantized_values[i] = 0;
//     }
//     sum     = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
//     amax    = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<float>());
//     float d = amax == 0 ? 1 : amax / 127;
//
// #pragma unroll(ElementsPerWI)
//     for (int i = 0; i < ElementsPerWI; i++) {
//         quantized_values[i] = sycl::round(wi_f32_vals[i] / d);
//     }
//
//     d = amax == 0 ? 0 : d;
//
//     auto sblock_offset = ((QK8_1 * subgroup_id) / QK_K) * QK_K;
//     auto quant_ptr     = (int8_t *) ((char *) reordered_q8_tensor + row_offset + sblock_offset);
//
//     // Reorder offsets to get per blockload interleave (values are 16 ints apart inside each superblock)
//     constexpr auto block_load_width = WARP_SIZE * sizeof(int);
//     // (subgroup_id % (QK_K / QK8_1))            -> Starting destination for subgroup
//     // (WARP_SIZE * ElementsPerWI) / sizeof(int) -> number of loads per wi in a superblock
//     auto           sg_offset        = (subgroup_id % (QK_K / QK8_1)) * (WARP_SIZE * ElementsPerWI) / sizeof(int);
//     // (subgroup_id / 8) -> Which half of the subgroup I am
//     // sizeof(int)       -> Number of values required for dp4a
//     // (block_load_width * (wi_id / 2)) % QK_K -> Offset to ensure block loads load contiguous data.
//     // ElementsPerWI * (wi_id % 2)   -> WI location the designed block
//     auto           chunk_offset =
//         (wi_id / 8) * sizeof(int) + (block_load_width * (wi_id / 2)) % QK_K + ElementsPerWI * (wi_id % 2);
//
//     *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(quant_ptr + sg_offset + chunk_offset) = quantized_values;
//
//     if (wi_id == 0) {
//         *ds_ptr = sycl::half2(sycl::half(d), sycl::half(sum));
//     }
// }

template <int ElementsPerWI>
__dpct_inline__ void quantize_and_reorder_q8_1_soa(const float * __restrict__ x, void * reordered_q8_tensor,
                                                   const int kx, const int kx_padded, const sycl::nd_item<1> & it) {
    /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */

    auto subgroup_id = it.get_group(0);
    auto wi_id       = it.get_local_id(0);

    const int num_blocks_per_row = kx / QK8_1;
    auto      row                = subgroup_id / num_blocks_per_row;
    auto      col                = subgroup_id % num_blocks_per_row;

    auto row_offset = row * (kx_padded / QK8_1) * sizeof(block_q8_1);
    auto col_offset = QK8_1 * col + wi_id * ElementsPerWI;

    auto quant_ptr = (int8_t *) ((char *) reordered_q8_tensor + row_offset + col_offset);
    auto ds_ptr    = (sycl::half2 *) ((char *) reordered_q8_tensor + row_offset + kx + col * sizeof(sycl::half2));

    sycl::vec<float, ElementsPerWI>  wi_f32_vals;
    sycl::vec<int8_t, ElementsPerWI> quantized_values;

    auto float_ptr_offset = subgroup_id * QK8_1 + ElementsPerWI * wi_id;
    wi_f32_vals           = *reinterpret_cast<const sycl::vec<float, ElementsPerWI> *>(x + float_ptr_offset);

    float sum  = 0.0f;
    float amax = 0.0f;

#pragma unroll(ElementsPerWI)
    for (int i = 0; i < ElementsPerWI; i++) {
        sum += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_values[i] = 0;
    }
    sum     = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
    amax    = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<float>());
    float d = amax == 0 ? 1 : amax / 127;

#pragma unroll(ElementsPerWI)
    for (int i = 0; i < ElementsPerWI; i++) {
        quantized_values[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;

    *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(quant_ptr) = quantized_values;
    if (wi_id == 0) {
        *ds_ptr = sycl::half2(sycl::half(d), sycl::half(sum));
    }
}
