/***************************************************************************
 *
 *  Copyright (C) 2025 Codeplay Software Ltd.
 *  Copyright (C) 2025 Intel Corporation
 *
 *  MIT License
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  quantize.hpp
 *
 *  Description:
 *     Sycl backend specific quantization functions
 **************************************************************************/

#pragma once

#include <sycl/nd_item.hpp>
#include "ggml-sycl/dpct/helper.hpp"

typedef void (*sycl_quantize_t)(const float * x, void * vy, const int kx, const int kx_padded,
                                const sycl::nd_item<3> & item_ct1);

template <int QUANT_BLOCK_TILE>
void quantize_q8_1(const float * x, void * vy, int kx, int kx_padded, const sycl::nd_item<3> & item_ct1);

template <int ElementsPerWI>
__dpct_inline__ void quantize_and_reorder_q8_1_linear(const float * x, void * reordered_q8_tensor, int kx,
                                                      int kx_padded, const sycl::nd_item<1> & it);

template <int ElementsPerWI>
__dpct_inline__ void quantize_and_reorder_q8_1_soa(const float * x, void * reordered_q8_tensor, int kx, int kx_padded,
                                                   const sycl::nd_item<1> & it);
