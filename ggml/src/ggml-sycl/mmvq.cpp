#include "mmvq.hpp"

#include "ggml.h"
#include "common.hpp"
#include "quants.hpp"
#include "vecdotq.hpp"

template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_reorder(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                                  const int ncols, const int nrows, const sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;


    // int32_t* ptr_a = const_cast<int32_t *>(reinterpret_cast<const int32_t*>(vx));
    // int32_t* wtr_b = const_cast<int32_t *>(reinterpret_cast<const int32_t*>(vy));
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

    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  row          = workgroup_id * sg_range + sg_id;

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum = 0.0f;


    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const auto         bx_offset      = block_type::get_block_offset(ibx, nblocks);
        const auto         d_offset       = block_type::get_d_offset(nrows, ncols, ibx);
        // Y block index that aligns with ibx
        const int iby = i * block_type::block_to_q8_1_ratio();
        const int8_t* q8_1_quant_ptr = (const int8_t*)vy + iby * QK8_1;
        const sycl::half2* q8_1_ds_ptr = (const sycl::half2*)((const char*)vy + ncols + iby * sizeof(sycl::half2));

        if (syclcompat::global_id::x() > -1 && syclcompat::global_id::x() < 16) {
            sycl::ext::oneapi::experimental::printf("sgid %d, %d, %zu, %zu\n", syclcompat::global_id::x(), ibx, bx_offset, d_offset);
        }
        if (syclcompat::global_id::x() > 239 && syclcompat::global_id::x() < 256) {
            sycl::ext::oneapi::experimental::printf("sgid %d, %d, %zu, %zu\n", syclcompat::global_id::x(), ibx, bx_offset, d_offset);
        }

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            // x block quant index when casting the quants to int
            const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);

            partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }

        if (syclcompat::global_id::x() > 239 && syclcompat::global_id::x() < 256) {
            sycl::ext::oneapi::experimental::printf("sgid %d, %.8f\n", syclcompat::global_id::x(), partial_sum);
        }
    }

    auto sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum, std::plus<>());


    if (sg.leader()) {
        dst[row] = sum;
        sycl::ext::oneapi::experimental::printf("row %d %.8f \n", row, dst[row]);
    }

    const int8_t* ptr = reinterpret_cast<const int8_t*>(vy);
    const int8_t* wtr = reinterpret_cast<const int8_t*>(vx);
    auto gid = nd_item.get_global_linear_id();
    if (gid == 0) {
        for (auto i = 0; i < nrows * ncols / 2; i++) {
            sycl::ext::oneapi::experimental::printf("%4d ", wtr[i]);
            if (i % 16 == 15) sycl::ext::oneapi::experimental::printf("\n");
            if (i % 128 == 127) sycl::ext::oneapi::experimental::printf("\n");
        }
        // sycl::ext::oneapi::experimental::printf("\n");
        for (auto i = 0; i < ncols; i++) {
            // sycl::ext::oneapi::experimental::printf("%4d ", ptr[i]);
            // if (i % 16 == 15)sycl::ext::oneapi::experimental::printf("\n");
        }
        // sycl::ext::oneapi::experimental::printf("\n");
        for (auto i = 0; i < ((ncols / QK8_1) * 4); i++) {
            // sycl::ext::oneapi::experimental::printf("%4d ", ptr[ncols + i]);
            // if (i % 16 == 15)sycl::ext::oneapi::experimental::printf("\n");
        }
        sycl::ext::oneapi::experimental::printf("\n");
    }

}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                          const int ncols, const int nrows, const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;  // Ensuring blocks_per_warp > 0

    assert(blocks_per_warp > 0);

    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            const int iqs = elem + vdr * (item_ct1.get_local_id(2) %
                                          (qi / vdr));  // x block quant index when casting the quants to int

            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xxs_q8_1(const void *__restrict__ vx,
                                       const void *__restrict__ vy,
                                       float *__restrict__ dst, const int ncols,
                                       const int nrows,
                                       const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);

// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xxs_q8_1(&x[ibx], &y[iby], iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xs_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid, ksigns64);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_s_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_xxs_q8_1(const void *__restrict__ vx,
                                       const void *__restrict__ vy,
                                       float *__restrict__ dst, const int ncols,
                                       const int nrows,
                                       const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid, ksigns64);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq1_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_s_q8_1(&x[ibx], &y[iby], iqs, iq1s_grid_gpu);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq1_m_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_m_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq4_nl_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_nl_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}


template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq4_xs_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_xs_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

static void reorder_mul_mat_vec_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, (block_num_y * WARP_SIZE));
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

static void mul_mat_vec_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

static void mul_mat_vec_q4_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK4_0, QI4_1, block_q4_1,
                                      VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_q5_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK5_0, QI5_0, block_q5_0,
                                      VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_q5_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK5_1, QI5_1, block_q5_1,
                                      VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_q8_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK8_0, QI8_0, block_q8_0,
                                      VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_q2_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI2_K, block_q2_K,
                                      VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_q3_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI3_K, block_q3_K,
                                      VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_q4_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI4_K, block_q4_K,
                                      VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void reorder_mul_mat_vec_q4_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_item);
                            });
    }).wait();
}


static void mul_mat_vec_q5_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI5_K, block_q5_K,
                                      VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void reorder_mul_mat_vec_q6_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}
static void mul_mat_vec_q6_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI6_K, block_q6_K,
                                      VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}


static void mul_mat_vec_iq2_xxs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_xxs_q8_1<QK_K, QI2_XXS/2, block_iq2_xxs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq2_xs_q8_1_sycl(const void *vx, const void *vy,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_xs_q8_1<QK_K, QI2_XS/2, block_iq2_xs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq2_s_q8_1_sycl(const void *vx, const void *vy,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_s_q8_1<QK_K, QI2_S/2, block_iq2_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq3_xxs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq3_xxs_q8_1<QK_K, QI3_XXS/2, block_iq3_xxs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq3_s_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq3_s_q8_1<QK_K, QI3_S/2, block_iq3_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq1_s_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq1_s_q8_1<QK_K, QI1_S, block_iq1_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq1_m_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq1_m_q8_1<QK_K, QI1_S, block_iq1_m, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq4_nl_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_NL == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq4_nl_q8_1<QK4_NL, QI4_NL, block_iq4_nl, 2>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq4_xs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq4_xs_q8_1<QK_K, QI4_XS/4, block_iq4_xs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

void ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low,
                                const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_col_size,
                                const dpct::queue_ptr & stream) {
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne00     = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    int id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;
    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into

    for (int i = 0; i < src1_ncols; i++) {
        const size_t src1_ddq_i_offset = i * src1_padded_col_size * q8_1_ts / q8_1_bs;
        const char * src1_ddq_i_bs     = src1_ddq_i + src1_ddq_i_offset;
        float *      dst_dd_i_bs       = dst_dd_i + i * dst->ne[0];
        switch (src0->type) {
            case GGML_TYPE_Q4_0:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl\n");
                    reorder_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                } else {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl\n");
                    mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q4_1:
                mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q5_0:
                mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q5_1:
                mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q8_0:
                mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q2_K:
                mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q3_K:
                mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q4_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                    reorder_mul_mat_vec_q4_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                } else {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                    mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q5_K:
                mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q6_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                    reorder_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                } else {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_k_q8_1_sycl\n");
                    mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_IQ1_S:
                mul_mat_vec_iq1_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ1_M:
                mul_mat_vec_iq1_m_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_XXS:
                mul_mat_vec_iq2_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_XS:
                mul_mat_vec_iq2_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_S:
                mul_mat_vec_iq2_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ3_XXS:
                mul_mat_vec_iq3_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ3_S:
                mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_NL:
                mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_XS:
                mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            default:
                GGML_ABORT("fatal error");
        }
    }
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(ctx);
}
