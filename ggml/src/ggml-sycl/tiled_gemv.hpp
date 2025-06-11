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
 *  tiled_gemv.hpp
 *
 *  Description:
 *     Tiled gemv public API
 **************************************************************************/

#pragma once

#include <sycl/nd_item.hpp>

#include "dpct/helper.hpp"

void q4_K_q8_1_tiled_gemv(const void * weights, const void * input, float * dst, int nrows, int ncols,
                          const sycl::nd_item<1> & it);

void mul_mat_q4_K_q8_1_tiled_gemv(const void * vx, const void * vy, float * dst, int ncols, int nrows,
                                  dpct::queue_ptr stream);
