//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cmath>
#include <cstddef>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void reorg_yolo(const T* arg, T* out, const Shape& in_shape, int64_t stride)
            {
                // [N, C, H, W]
                size_t in_N = in_shape[0];
                size_t in_C = in_shape[1];
                size_t in_H = in_shape[2];
                size_t in_W = in_shape[3];

                // Inferce output shape logic:
                // in_shape [N,C,H,W] -> out_shape [N, C*(stride*stride), H/stride, W/stride]
                // ReorgYolo imlementation calculates indexes like for backward:
                // in_shape [N,C,H,W] -> out_shape [N, C/(stride*stride), H*stride, W*stride]

                size_t impl_out_C = in_C / (stride * stride);
                if (impl_out_C == 0)
                {
                    throw ngraph_error(
                        "ReorgYolo. For [N, C, H, W] input shape, C >= (stride*stride) is "
                        "required.");
                }
                size_t impl_out_H = in_H * stride;
                size_t impl_out_W = in_W * stride;

                for (size_t n = 0; n < in_N; ++n)
                {
                    for (size_t c = 0; c < in_C; ++c)
                    {
                        for (size_t h = 0; h < in_H; ++h)
                        {
                            for (size_t w = 0; w < in_W; ++w)
                            {
                                size_t dest_index =
                                    n * in_C * in_H * in_W + c * in_H * in_W + h * in_W + w;

                                size_t impl_c = c % impl_out_C;
                                size_t offset = c / impl_out_C;

                                size_t impl_w = w * stride + offset % stride;
                                size_t impl_h = h * stride + offset / stride;
                                size_t arg_index = n * impl_w * impl_out_C * impl_out_H * impl_out_W +
                                                   impl_c * impl_out_H * impl_out_W +
                                                   impl_h * impl_out_W + impl_w;

                                out[dest_index] = arg[arg_index];
                            }
                        }
                    }
                }
            }
        }
    }
}
