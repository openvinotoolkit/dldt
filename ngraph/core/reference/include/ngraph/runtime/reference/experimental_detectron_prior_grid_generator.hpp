//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <cstddef>
#include <cstdint>
#include <ngraph/runtime/host_tensor.hpp>
#include <vector>
#include "ngraph/node.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void experimental_detectron_prior_grid_generator(const T* priors,
                                                             const Shape& priors_shape,
                                                             const Shape& feature_map_shape,
                                                             const Shape& im_data_shape,
                                                             T* output_rois,
                                                             int64_t grid_h,
                                                             int64_t grid_w,
                                                             float stride_h,
                                                             float stride_w)
            {
                const int64_t num_priors = static_cast<int64_t>(priors_shape[0]);
                const int64_t layer_width = grid_w ? grid_w : feature_map_shape[3];
                const int64_t layer_height = grid_h ? grid_h : feature_map_shape[2];
                const float step_w =
                    stride_w ? stride_w : static_cast<float>(im_data_shape[3]) / layer_width;
                const float step_h =
                    stride_h ? stride_h : static_cast<float>(im_data_shape[2]) / layer_height;

                for (int64_t h = 0; h < layer_height; ++h)
                {
                    for (int64_t w = 0; w < layer_width; ++w)
                    {
                        for (int64_t s = 0; s < num_priors; ++s)
                        {
                            output_rois[0] = priors[4 * s + 0] + step_w * (w + 0.5f);
                            output_rois[1] = priors[4 * s + 1] + step_h * (h + 0.5f);
                            output_rois[2] = priors[4 * s + 2] + step_w * (w + 0.5f);
                            output_rois[3] = priors[4 * s + 3] + step_h * (h + 0.5f);
                            output_rois += 4;
                        }
                    }
                }
            }

//             void experimental_detectron_prior_grid_generator(const float* priors,
//                                                              const Shape& priors_shape,
//                                                              const Shape& feature_map_shape,
//                                                              const Shape& im_data_shape,
//                                                              float* output_rois,
//                                                              int64_t grid_h,
//                                                              int64_t grid_w,
//                                                              float stride_h,
//                                                              float stride_w);
//
//             void experimental_detectron_prior_grid_generator_postprocessing(
//                 const HostTensorVector& outputs,
//                 const ngraph::element::Type output_type,
//                 const std::vector<float>& output_rois,
//                 const Shape& output_rois_shape);
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
