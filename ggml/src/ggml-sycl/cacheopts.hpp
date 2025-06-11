#ifndef GGML_SYCL_CACHEOPTS_HPP
#define GGML_SYCL_CACHEOPTS_HPP

enum LSC_LDCC {
    LSC_LDCC_DEFAULT   = 0,
    LSC_LDCC_L1UC_L3UC = 1,  // Override to L1 uncached and L3 uncached
    LSC_LDCC_L1UC_L3C  = 2,  // Override to L1 uncached and L3 cached
    LSC_LDCC_L1C_L3UC  = 3,  // Override to L1 cached and L3 uncached
    LSC_LDCC_L1C_L3C   = 4,  // Override to L1 cached and L3 cached
    LSC_LDCC_L1S_L3UC  = 5,  // Override to L1 streaming load and L3 uncached
    LSC_LDCC_L1S_L3C   = 6,  // Override to L1 streaming load and L3 cached
    LSC_LDCC_L1IAR_L3C = 7,  // Override to L1 invalidate-after-read, and L3 cached
};

// Store message caching control (also used for atomics)
enum LSC_STCC {
    LSC_STCC_DEFAULT   = 0,
    LSC_STCC_L1UC_L3UC = 1,  // Override to L1 uncached and L3 uncached
    LSC_STCC_L1UC_L3WB = 2,  // Override to L1 uncached and L3 written back
    LSC_STCC_L1WT_L3UC = 3,  // Override to L1 written through and L3 uncached
    LSC_STCC_L1WT_L3WB = 4,  // Override to L1 written through and L3 written back
    LSC_STCC_L1S_L3UC  = 5,  // Override to L1 streaming and L3 uncached
    LSC_STCC_L1S_L3WB  = 6,  // Override to L1 streaming and L3 written back
    LSC_STCC_L1WB_L3WB = 7,  // Override to L1 written through and L3 written back
};

#endif
