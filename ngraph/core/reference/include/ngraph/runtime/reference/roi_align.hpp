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

#include <algorithm>
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/op/roi_align.hpp" // for ROIAlign:PoolingMode
#include "ngraph/shape.hpp"
namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            using ROIPoolingMode = op::v3::ROIAlign::PoolingMode;
            template <typename T>
            void roi_align(const T* feature_maps,
                           const T* rois,
                           const int64_t* batch_indices,
                           T* out,
                           const Shape& feature_maps_shape,
                           const Shape& rois_shape,
                           const Shape& batch_indices_shape,
                           const Shape& out_shape,
                           const int pooled_height,
                           const int pooled_width,
                           const int sampling_ratio,
                           const float spatial_scale,
                           const ROIPoolingMode& pooling_mode)
            {
                auto C = feature_maps_shape[1];
                auto feature_map_height = feature_maps_shape[2];
                auto feature_map_width = feature_maps_shape[3];
                auto num_rois = rois_shape[0];

                CoordinateTransform feature_maps_transform(feature_maps_shape);
                CoordinateTransform rois_transform(rois_shape);
                CoordinateTransform out_transform(out_shape);

                for (uint roi_index = 0; roi_index < num_rois; roi_index++)
                {
                    // Get ROI`s corners
                    T x1 = rois[rois_transform.index({roi_index, 0})] * spatial_scale;
                    T y1 = rois[rois_transform.index({roi_index, 1})] * spatial_scale;
                    T x2 = rois[rois_transform.index({roi_index, 2})] * spatial_scale;
                    T y2 = rois[rois_transform.index({roi_index, 3})] * spatial_scale;

                    T roi_width = std::max(x2 - x1, static_cast<T>(1.0));
                    T roi_height = std::max(y2 - y1, static_cast<T>(1.0));

                    T bin_width = roi_width / pooled_width;
                    T bin_height = roi_height / pooled_height;

                    uint64_t num_samples_in_bin = sampling_ratio * sampling_ratio;

                    T sample_distance_x = bin_width / static_cast<T>(sampling_ratio);
                    T sample_distance_y = bin_height / static_cast<T>(sampling_ratio);

                    std::vector<std::pair<uint64_t, uint64_t>> pooling_points;
                    std::vector<T> pooling_weights;

                    pooling_points.reserve(4 * num_samples_in_bin * pooled_height * pooled_width);
                    pooling_weights.reserve(4 * num_samples_in_bin * pooled_height * pooled_width);

                    // Save the sample coords and weights as they will be identical across all
                    // channels
                    for (uint y_bin_ind = 0; y_bin_ind < pooled_height; y_bin_ind++)
                    {
                        for (uint x_bin_ind = 0; x_bin_ind < pooled_width; x_bin_ind++)
                        {
                            for (uint y_sample_ind = 0; y_sample_ind < sampling_ratio;
                                 y_sample_ind++)
                            {
                                T sample_y = y1 + static_cast<T>(y_bin_ind) * bin_height +
                                             sample_distance_y * (static_cast<T>(y_sample_ind) +
                                                                  static_cast<T>(0.5f));

                                for (int64_t x_sample_ind = 0; x_sample_ind < sampling_ratio;
                                     x_sample_ind++)
                                {
                                    T sample_x = x1 + static_cast<T>(x_bin_ind) * bin_width +
                                                 sample_distance_x * (static_cast<T>(x_sample_ind) +
                                                                      static_cast<T>(0.5f));

                                    if (sample_x < -1.0 || sample_x > feature_map_width ||
                                        sample_y < -1.0 || sample_y > feature_map_height)
                                    {
                                        // For this sample we save 4x point (0,0) with weight 0
                                        pooling_points.insert(pooling_points.end(), 4, {0, 0});
                                        pooling_weights.insert(pooling_weights.end(), 4, {0});
                                        continue;
                                    }

                                    sample_x = std::max(sample_x, T{0});
                                    sample_y = std::max(sample_y, T{0});

                                    auto sample_y_low = static_cast<uint64_t>(sample_y);
                                    auto sample_x_low = static_cast<uint64_t>(sample_x);
                                    uint64_t sample_y_high;
                                    uint64_t sample_x_high;

                                    if (sample_y_low >= feature_map_height - 1)
                                    {
                                        sample_y_high = sample_y_low = feature_map_height - 1;
                                        sample_y = (T)sample_y_low;
                                    }
                                    else
                                    {
                                        sample_y_high = sample_y_low + 1;
                                    }

                                    if (sample_x_low >= feature_map_height - 1)
                                    {
                                        sample_x_high = sample_x_low = feature_map_width - 1;
                                        sample_x = (T)sample_x_low;
                                    }
                                    else
                                    {
                                        sample_x_high = sample_x_low + 1;
                                    }
                                    pooling_points.push_back({sample_y_low, sample_x_low});
                                    pooling_points.push_back({sample_y_low, sample_x_high});
                                    pooling_points.push_back({sample_y_high, sample_x_low});
                                    pooling_points.push_back({sample_y_high, sample_x_high});

                                    // weight calculation for bilinear interpolation
                                    auto ly = sample_y - static_cast<T>(sample_y_low);
                                    auto lx = sample_x - static_cast<T>(sample_x_low);
                                    auto hy = static_cast<T>(1.) - ly;
                                    auto hx = static_cast<T>(1.) - lx;

                                    pooling_weights.push_back(hy * hx);
                                    pooling_weights.push_back(hy * lx);
                                    pooling_weights.push_back(ly * hx);
                                    pooling_weights.push_back(ly * lx);
                                }
                            }
                        }
                    }

                    std::vector<T> tmp_out;

                    for (uint channel_index = 0; channel_index < C; channel_index++)
                    {
                        tmp_out.reserve(pooled_height * pooled_width);
                        uint sample_index = 0;
                        for (uint y_bin_ind = 0; y_bin_ind < pooled_height; y_bin_ind++)
                        {
                            for (uint x_bin_ind = 0; x_bin_ind < pooled_width; x_bin_ind++)
                            {
                                T pooled_value = 0;
                                for (uint bin_sample_ind = 0; bin_sample_ind < num_samples_in_bin;
                                     bin_sample_ind++)
                                {
                                    // the four parts are values of the four closest surrounding
                                    // neighbours of considered sample, then basing on all sampled
                                    // values in bin we calculate pooled value
                                    auto sample_part_1 = feature_maps[feature_maps_transform.index(
                                        {static_cast<uint64_t>(batch_indices[roi_index]),
                                         channel_index,
                                         pooling_points[sample_index].first,
                                         pooling_points[sample_index].second})];
                                    auto sample_part_2 = feature_maps[feature_maps_transform.index(
                                        {static_cast<uint64_t>(batch_indices[roi_index]),
                                         channel_index,
                                         pooling_points[sample_index + 1].first,
                                         pooling_points[sample_index + 1].second})];
                                    auto sample_part_3 = feature_maps[feature_maps_transform.index(
                                        {static_cast<uint64_t>(batch_indices[roi_index]),
                                         channel_index,
                                         pooling_points[sample_index + 2].first,
                                         pooling_points[sample_index + 2].second})];
                                    auto sample_part_4 = feature_maps[feature_maps_transform.index(
                                        {static_cast<uint64_t>(batch_indices[roi_index]),
                                         channel_index,
                                         pooling_points[sample_index + 3].first,
                                         pooling_points[sample_index + 3].second})];

                                    switch (pooling_mode)
                                    {
                                    case ROIPoolingMode::MAX:
                                    {
                                        T sample_value = std::max(
                                            {pooling_weights[sample_index] * sample_part_1,
                                             pooling_weights[sample_index + 1] * sample_part_2,
                                             pooling_weights[sample_index + 2] * sample_part_3,
                                             pooling_weights[sample_index + 3] * sample_part_4});

                                        pooled_value = sample_value > pooled_value ? sample_value
                                                                                   : pooled_value;
                                        break;
                                    }
                                    case ROIPoolingMode::AVG:
                                    default:
                                    {
                                        T sample_value =
                                            pooling_weights[sample_index] * sample_part_1 +
                                            pooling_weights[sample_index + 1] * sample_part_2 +
                                            pooling_weights[sample_index + 2] * sample_part_3 +
                                            pooling_weights[sample_index + 3] * sample_part_4;
                                        pooled_value += sample_value / (num_samples_in_bin);
                                    }
                                    }
                                    sample_index += 4;
                                }
                                tmp_out.push_back(pooled_value);
                            }
                        }
                        // save the calculations for all bins across this channel
                        uint64_t output_channel_offset =
                            out_transform.index({static_cast<uint64_t>(roi_index),
                                                 static_cast<uint64_t>(channel_index),
                                                 static_cast<uint64_t>(0),
                                                 static_cast<uint64_t>(0)});
                        std::copy(tmp_out.begin(), tmp_out.end(), out + output_channel_offset);

                        tmp_out.clear();
                    }
                }
                return;
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
