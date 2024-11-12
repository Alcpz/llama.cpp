#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_fp16.h>

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

#define QR5_K 2
#define QI5_K (QK_K / (4 * QR5_K))

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

typedef struct {
  half2 dm;                     // super-block scale for quantized scales/mins
  uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
  uint8_t qh[QK_K / 8];         // quants, high bit
  uint8_t qs[QK_K / 2];         // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) ==
                  2 * sizeof(ggml_half) + K_SCALE_SIZE + QK_K / 2 + QK_K / 8,
              "wrong q5_K block size/padding");

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

#define QK8_0 32
#define QR8_0 1
#define QI8_0 (QK8_0 / (4 * QR8_0))
typedef struct {
  half d;           // delta
  int8_t qs[QK8_0]; // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0,
              "wrong q8_0 block size/padding");

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

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int *__restrict__ v, const int *__restrict__ u,
    const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
    const half2 &dm4, const float *__restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;

        const int dot1 =
            __dp4a(v1i, u[2 * i + 1],
                   __dp4a(v0i, u[2 * i + 0], 0)); // SIMD dot product
        const int dot2 =
            __dp4a(0x01010101, u[2 * i + 1],
                   __dp4a(0x01010101, u[2 * i + 0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m +=
            d8[i] *
            (dot2 *
             m[i]); // multiply constant part of q4_K with sum of q8_1 values
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

static __device__ __forceinline__ float
vec_dot_q4_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_q4_K *bq4_K = (const block_q4_K *)vbq;

    int v[2];
    int u[2 * QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));

    // printf("\n  vdq_q4k(%d, %d, %d, %d) = bq8_offset = %#010x", blockIdx.y,
    //     blockIdx.x, threadIdx.y, threadIdx.x, bq8_offset);

    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

    const int *q4 =
        (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t *scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] =
            ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] =
            ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uint8_t *sc = (const uint8_t *)aux;
    const uint8_t *m = sc + 2;

    // printf("\n  vdq_q4k(%d, %d, %d, %d) = V %#010x, %#010x", blockIdx.y,
    //    blockIdx.x, threadIdx.y, threadIdx.x, v[0], v[1]);

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 *bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2half(bq8i->ds);

        const int *q8 = (const int *)bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }

    // printf("\n  vdq_q4k(%d, %d, %d, %d) = U %#010x, %#010x, %#010x, %#010x",
    // blockIdx.y,
    //        blockIdx.x, threadIdx.y, threadIdx.x, u[0], u[1], u[2], u[3]);

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
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

static __device__ __forceinline__ int get_int_from_int8(const int8_t *x8,
                                                        const int &i32) {
  const uint16_t *x16 =
      (const uint16_t *)(x8 +
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
    const int row = blockIdx.x * blockDim.y + threadIdx.y;

    if (row >= nrows) {
        return;
    }

    // printf("\nROW(%d, %d, %d, %d) = %d", blockIdx.y, blockIdx.x, threadIdx.y,
    // threadIdx.x, row);

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;

    // partial sum for each thread
    // printf("\nBPR_W(%d, %d, %d, %d) = %d, %d", blockIdx.y, blockIdx.x,
    // threadIdx.y, threadIdx.x, blocks_per_row, blocks_per_warp);

    float tmp = 0.0f;

    const block_q_t *x = (const block_q_t *)vx;
    const block_q8_1 *y = (const block_q8_1 *)vy;

    for (int i = threadIdx.x / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i; // x block index

        const int iby = i * (qk / QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (threadIdx.x %
             (qi / vdr)); // x block quant index when casting the quants to int

        // printf("\n  vdqcuda_call(%d)(%d, %d, %d, %d) = %d, %d, %d", i,
        // blockIdx.y, blockIdx.x,  threadIdx.y, threadIdx.x, ibx, iby, iqs);
        tmp += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
        // printf("\n  vdqcuda_call(%d)(%d, %d, %d, %d) = %.8f", i, blockIdx.y,
        // blockIdx.x,  threadIdx.y, threadIdx.x, tmp);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        // printf("\n* pre_reduce(%d)(%d, %d, %d, %d) = %.8f", mask, blockIdx.y,
        // blockIdx.x, threadIdx.y, threadIdx.x, tmp);
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
        // // printf("\nPOST_REDUCE(%d)(%d, %d, %d, %d) = %.8f", mask,
        // blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, tmp);
    }

    if (threadIdx.x == 0) {
        // printf("\nTMP(%d, %d, %d, %d) = %d, %.8f", blockIdx.y, blockIdx.x,
        // threadIdx.y, threadIdx.x, row, tmp);
        dst[row] = tmp;
    }
}


static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int *__restrict__ vl, const int *__restrict__ vh,
    const int *__restrict__ u, const uint8_t *__restrict__ sc,
    const uint8_t *__restrict__ m, const half2 &dm5,
    const float *__restrict__ d8) {

#if __CUDA_ARCH__ >=                                                           \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
  float sumf_d = 0.0f;
  float sumf_m = 0.0f;

#pragma unroll
  for (int i = 0; i < QR5_K; ++i) {
    const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
    const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;

    const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
    const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

    const int v0i = vl0i | vh0i;
    const int v1i = vl1i | vh1i;

    const int dot1 = __dp4a(v0i, u[2 * i + 0],
                            __dp4a(v1i, u[2 * i + 1], 0)); // SIMD dot product
    const int dot2 = __dp4a(0x01010101, u[2 * i + 0],
                            __dp4a(0x01010101, u[2 * i + 1], 0)); // sum of u

    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
  }

  const float2 dm5f = __half22float2(dm5);

  return dm5f.x * sumf_d - dm5f.y * sumf_m;

#else
  assert(false);
  return 0.0f;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}


static __device__ __forceinline__ float
vec_dot_q5_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

#ifndef GGML_QKK_64
  const block_q5_K *bq5_K = (const block_q5_K *)vbq;

  int vl[2];
  int vh[2];
  int u[2 * QR5_K];
  float d8[QR5_K];

  const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
  const int *ql =
      (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  const int *qh = (const int *)(bq5_K->qh + 4 * ((iqs / 2) % 4));

  vl[0] = ql[0];
  vl[1] = ql[4];

  vh[0] = qh[0] >> bq8_offset;
  vh[1] = qh[4] >> bq8_offset;

  const uint16_t *scales = (const uint16_t *)bq5_K->scales;
  uint16_t aux[2];
  const int j = bq8_offset / 2;
  if (j < 2) {
    aux[0] = scales[j + 0] & 0x3f3f;
    aux[1] = scales[j + 2] & 0x3f3f;
  } else {
    aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
    aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
  }
  const uint8_t *sc = (const uint8_t *)aux;
  const uint8_t *m = sc + 2;

#pragma unroll
  for (int i = 0; i < QR5_K; ++i) {
    const block_q8_1 *bq8i = bq8_1 + bq8_offset + i;
    d8[i] = __low2float(bq8i->ds);

    const int *q8 = (const int *)bq8i->qs + ((iqs / 2) % 4);
    u[2 * i + 0] = q8[0];
    u[2 * i + 1] = q8[4];
  }

  return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);

#else
  NO_DEVICE_CODE;
#endif
}

#define VDR_Q8_0_Q8_1_MMVQ 2
template <int vdr>
static __device__ __forceinline__ float
vec_dot_q8_0_q8_1_impl(const int *v, const int *u, const float &d8_0,
                       const float &d8_1) {

#if __CUDA_ARCH__ >=                                                           \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
  int sumi = 0;

#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    // SIMD dot product of quantized values
    sumi = __dp4a(v[i], u[i], sumi);
  }

  return d8_0 * d8_1 * sumi;
#else
  assert(false);
  return 0.0f;
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

static __device__ __forceinline__ float
vec_dot_q8_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

  const block_q8_0 *bq8_0 = (const block_q8_0 *)vbq;

  int v[VDR_Q8_0_Q8_1_MMVQ];
  int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
  for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
    u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
  }

  return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d,
                                                    __low2half(bq8_1->ds));
}

#define GGML_CUDA_MMV_Y 1

#define VDR_Q4_K_Q8_1_MMVQ 2

void mul_mat_vec_q4_K_q8_1_sycl_facade(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows, CUstream stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q_interop<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                          vec_dot_q4_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

#define VDR_Q6_K_Q8_1_MMVQ 1
void mul_mat_vec_q6_K_q8_1_sycl_facade(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows, CUstream stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q_interop<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                          vec_dot_q6_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

#define VDR_Q5_K_Q8_1_MMVQ 2
void mul_mat_vec_q5_K_q8_1_sycl_facade(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows, CUstream stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q_interop<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                          vec_dot_q5_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

void mul_mat_vec_q8_0_q8_1_sycl_facade(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows, CUstream stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, 1, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q_interop<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                          vec_dot_q8_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static __global__ void quantize_q8_1(const float *__restrict__ x,
                                     void *__restrict__ vy, const int kx,
                                     const int kx_padded) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;

    if (ix >= kx_padded) {
        return;
    }

    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    const int i_padded = iy * kx_padded + ix;

    block_q8_1 *y = (block_q8_1 *)vy;

    const int ib = i_padded / QK8_1;  // block index
    const int iqs = i_padded % QK8_1; // quant index

    const float xi = ix < kx ? x[iy * kx + ix] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
        sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
    }

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    reinterpret_cast<half &>(y[ib].ds.x) = d;
    reinterpret_cast<half &>(y[ib].ds.y) = sum;
}

#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define CUDA_DEQUANTIZE_BLOCK_SIZE 256
void quantize_row_q8_1_facade(const float *x, void *vy, const int kx,
                              const int ky, const int kx_padded,
                              CUstream stream) {
    const int block_num_x =
        (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ky, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, kx, kx_padded);
}
