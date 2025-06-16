#ifndef GGML_SYCL_BUILTINS_HPP
#define GGML_SYCL_BUILTINS_HPP

#include <sycl/marray.hpp>

#include "cacheopts.hpp"

#define GGML_SYCL_UNREACHABLE(x) \
    assert(0 && x);              \
    printf(x);

#ifdef __SYCL_DEVICE_ONLY__
template <class T, int N> using vector_t = T __attribute__((ext_vector_type(N)));
#else
template <class T, int N> using vector_t = sycl::marray<T, N>;
#endif

#ifdef __SYCL_DEVICE_ONLY__
#    define SYCL_DEVICE_BUILTIN(x) SYCL_EXTERNAL extern "C" x
#else
#    define SYCL_DEVICE_BUILTIN(x)                                                      \
        inline x {                                                                      \
            GGML_SYCL_UNREACHABLE("Attempting to use a device built-in in host code."); \
        }
#endif

#ifdef __SYCL_DEVICE_ONLY__
#    define SYCL_DEVICE_OCL(x) SYCL_EXTERNAL extern "C" x
#else
#    define SYCL_DEVICE_OCL(x)
#endif

using uint32 = vector_t<uint, 32>;
using uint16 = vector_t<uint, 16>;
using uint8  = vector_t<uint, 8>;
using uint2  = vector_t<uint, 2>;

using short16  = vector_t<short, 16>;
using uint8_32 = vector_t<uint8_t, 32>;

// loads
SYCL_DEVICE_BUILTIN(uint16 __builtin_IB_subgroup_block_read_flat_u32_m16k16v1(intptr_t baseoffset, int width_minus_one,
                                                                              int height_minus_one, int pitch_minus_one,
                                                                              uint2 coord));

SYCL_DEVICE_BUILTIN(uint32_t __builtin_IB_subgroup_block_read_flat_u32_m1k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, uint2 coord));

//stores
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u32_m1k16v1(intptr_t baseoffset, int width_minus_one,
                                                                            int height_minus_one, int pitch_minus_one,
                                                                            uint2 coord, uint2 data));

// prefetches
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u32_m16k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, uint2 coord,
    LSC_LDCC cache_control));

SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u32_m1k16v1(intptr_t baseoffset, int width_minus_one,
                                                                               int height_minus_one,
                                                                               int pitch_minus_one, uint2 coord,
                                                                               LSC_LDCC cache_control));

#undef SYCL_DEVICE_BUILTIN
#undef SYCL_DEVICE_OCL

#endif
