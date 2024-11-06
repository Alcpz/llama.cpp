#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cuda.h>

#define WARP_SIZE 32
#define MIN_CC_DP4A 610

// QK = number of values after dequantization
// QK_K = super-block size

#define QK_K 256
#define K_SCALE_SIZE 12

// QR = QK / number of values before dequantization
// QI = number of 32 bit integers before dequantization

#define QI8_1 (QK8_1 / (4 * QR8_1))
#define QR8_1 1

#define QI4_K (QK_K / (4 * QR4_K))
#define QR4_K 2

#define QI6_K (QK_K / (4 * QR6_K))
#define QR6_K 2

typedef half ggml_half;
typedef half2 ggml_half2;

// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
typedef struct {
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR;
        ggml_half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K / 2];         // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) ==
                  2 * sizeof(ggml_half) + K_SCALE_SIZE + QK_K / 2,
              "wrong q4_K block size/padding");

// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct {
    uint8_t ql[QK_K / 2];     // quants, lower 4 bits
    uint8_t qh[QK_K / 4];     // quants, upper 2 bits
    int8_t scales[QK_K / 16]; // scales, quantized with 8 bits
    ggml_half d;              // super-block scale
} block_q6_K;
static_assert(sizeof(block_q6_K) ==
                  sizeof(ggml_half) + QK_K / 16 + 3 * QK_K / 4,
              "wrong q6_K block size/padding");

#define QK8_1 32
typedef struct {
    union {
        struct {
            ggml_half d; // delta
            ggml_half s; // d * sum(qs[i])
        } GGML_COMMON_AGGR;
        ggml_half2 ds;
    };
    int8_t qs[QK8_1]; // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2 * sizeof(ggml_half) + QK8_1,
              "wrong q8_1 block size/padding");

static __device__ __forceinline__ int get_int_b2(const void *x,
                                                 const int &i32) {
    const uint16_t *x16 =
        (const uint16_t *)x; // assume at least 2 byte alignment

    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_b4(const void *x,
                                                 const int &i32) {
    return ((const int *)x)[i32]; // assume at least 4 byte alignment
}

static __device__ __forceinline__ float
vec_dot_q4_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

#if __CUDA_ARCH__ <                                                            \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
    assert(false);
#endif
    const block_q4_K *bq4_K = (const block_q4_K *)vbq;

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    uint16_t aux16[2];
    const uint8_t *s = (const uint8_t *)aux16;

    const uint16_t *a = (const uint16_t *)bq4_K->scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    const float dall = bq4_K->dm[0];
    const float dmin = bq4_K->dm[1];

    const float d8_1 = __low2float(bq8_1[0].ds);
    const float d8_2 = __low2float(bq8_1[1].ds);

    const int ui1 = *((const int *)bq8_1[0].qs + (iqs / 2));
    const int ui2 = *((const int *)bq8_1[0].qs + (iqs / 2) + 4);
    const int ui3 = *((const int *)bq8_1[1].qs + (iqs / 2));
    const int ui4 = *((const int *)bq8_1[1].qs + (iqs / 2) + 4);

    const int *q4 = (const int *)bq4_K->qs + (iqs / 2);
    const int v1 = q4[0];
    const int v2 = q4[4];

    const int dot1 =
        __dp4a(ui2, v2 & 0x0f0f0f0f, __dp4a(ui1, v1 & 0x0f0f0f0f, 0));
    const int dot2 = __dp4a(ui4, (v2 >> 4) & 0x0f0f0f0f,
                            __dp4a(ui3, (v1 >> 4) & 0x0f0f0f0f, 0));
    const int dot3 = __dp4a(0x01010101, ui2, __dp4a(0x01010101, ui1, 0));
    const int dot4 = __dp4a(0x01010101, ui4, __dp4a(0x01010101, ui3, 0));

    sumf_d += d8_1 * (dot1 * s[0]) + d8_2 * (dot2 * s[1]);
    sumf_m += d8_1 * (dot3 * s[2]) + d8_2 * (dot4 * s[3]);

    return dall * sumf_d - dmin * sumf_m;
}

static __device__ __forceinline__ int get_int_from_uint8(const uint8_t *x8,
                                                         const int &i32) {
    const uint16_t *x16 =
        (uint16_t *)(x8 +
                     sizeof(int) * i32); // assume at least 2 byte alignment

    int x32 = 0;
    x32 |= x16[0] << 0;
    x32 |= x16[1] << 16;

    return x32;
}

static __device__ __forceinline__ int
get_int_from_int8_aligned(const int8_t *x8, const int &i32) {
    return *(
        (int *)(x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}

// contiguous v/x values
static __device__ __forceinline__ float
vec_dot_q6_K_q8_1_impl_mmvq(const int &vl, const int &vh,
                            const int *__restrict__ u,
                            const int8_t *__restrict__ scales, const float &d,
                            const float *__restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4 * i];

        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;

        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;

        const int vi =
            __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32

        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d * sumf;
}

static __device__ __forceinline__ float
vec_dot_q6_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_q6_K *bq6_K = (const block_q6_K *)vbq;

    const int bq8_offset =
        2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
    const int scale_offset =
        (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
    const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));

    const int vl = get_int_from_uint8(bq6_K->ql, iqs);
    const int vh =
        get_int_from_uint8(bq6_K->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) +
                                          iqs % (QI6_K / 4)) >>
        vh_shift;

    const int8_t *scales = bq6_K->scales + scale_offset;

    int u[QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i] = get_int_from_int8_aligned(bq8_1[bq8_offset + 2 * i].qs,
                                         iqs % QI8_1);
        d8[i] = __low2half(bq8_1[bq8_offset + 2 * i].ds);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}

typedef float (*vec_dot_q_cuda_t)(const void *__restrict__ vbq,
                                  const block_q8_1 *__restrict__ bq8_1,
                                  const int &iqs);

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void mul_mat_vec_q_interop(const void *__restrict__ vx,
                                             const void *__restrict__ vy,
                                             float *__restrict__ dst,
                                             const int ncols, const int nrows) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;

    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t *x = (const block_q_t *)vx;
    const block_q8_1 *y = (const block_q8_1 *)vy;

    for (int i = 0; i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i +
                        threadIdx.x / (qi / vdr); // x block index

        const int iby = (i + threadIdx.x / (qi / vdr)) *
                        (qk / QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (threadIdx.x %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

#define GGML_CUDA_MMV_Y 1

#define VDR_Q4_K_Q8_1_MMVQ 1
void mul_mat_vec_q4_K_q8_1_sycl_facade(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q_interop<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                          vec_dot_q4_K_q8_1>
        <<<block_nums, block_dims /*, 0, stream */>>>(vx, vy, dst, ncols,
                                                      nrows);
}

#define VDR_Q6_K_Q8_1_MMVQ 1
void mul_mat_vec_q6_K_q8_1_sycl_facade(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q_interop<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                          vec_dot_q6_K_q8_1>
        <<<block_nums, block_dims /* , 0, stream */>>>(vx, vy, dst, ncols,
                                                       nrows);
}
