// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/depth_to_space.hpp"
#include <cmath>
#include <numeric>
#include "ngraph/check.hpp"
#include "ngraph/runtime/opt_kernel/reshape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            bool depth_to_space(const char* data,
                                const Shape& data_shape,
                                char* out,
                                const Shape& out_shape,
                                const std::size_t block_size,
                                const op::DepthToSpace::DepthToSpaceMode mode,
                                const size_t elem_size)
            {
                const size_t n_dim = data_shape.at(0);
                const size_t c_dim = data_shape.at(1);
                const size_t spatial_dim_index = 2;
                const size_t spatial_dims = data_shape.size() - spatial_dim_index;
                const auto c_dim_divider = static_cast<int>(std::pow(block_size, spatial_dims));

                NGRAPH_CHECK(block_size > 0 && c_dim % c_dim_divider == 0,
                             "DepthToSpace: The input data's 'channels' axis size: ",
                             c_dim,
                             " must be a equivalent to ",
                             "'block_size'^'spatial_dims': ",
                             c_dim_divider);

                const size_t c_flat = c_dim / c_dim_divider;

                // First we have to disperse the data from depth channel, then rearrange them
                // so as appropriate chunks of data where close to their destination place.
                // Finally squeeze data from respective dimensions.
                Shape dispersed_shape{n_dim};
                for (size_t i = 0; i < spatial_dims; ++i)
                {
                    dispersed_shape.push_back(block_size);
                }
                for (size_t i = 0; i < spatial_dims; ++i)
                {
                    dispersed_shape.push_back(data_shape.at(spatial_dim_index + i));
                }
                std::vector<size_t> axes_order{0};
                switch (mode)
                {
                // x' = reshape(data, [N, C / (block_size ^ K), block_size, block_size, ...,
                //              block_size, D1, D2, ..., DK])
                // x'' = transpose(x',
                //                 [0,  1,  K + 2, 2, K + 3, 3, K + 4, 4, ..., K + (K + 1), K + 1])
                // y = reshape(x'',
                //             [N, C / (block_size ^ K), D1 * block_size, D2 * block_size,
                //             D3 * block_size, ..., DK * block_size])
                case op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST:
                {
                    dispersed_shape.insert(dispersed_shape.begin() + 1, c_flat);
                    axes_order.push_back(1);
                    for (size_t i = spatial_dim_index; i < data_shape.size(); ++i)
                    {
                        axes_order.push_back(spatial_dims + i);
                        axes_order.push_back(i);
                    }

                    break;
                }
                // x' = reshape(data, [N, block_size, block_size, ..., block_size,
                //              C / (block_size ^ K), D1, D2, ..., DK])
                // x'' = transpose(x', [0,  K + 1,  K + 2, 1, K + 3, 2, K + 4, 3, ...,
                //                 K + (K + 1), K])
                // y = reshape(x'', [N, C / (block_size ^ K), D1 * block_size, D2 * block_size,
                //             D3 * block_size, ..., DK * block_size])
                case op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST:
                {
                    dispersed_shape.insert(dispersed_shape.begin() + spatial_dims + 1, c_flat);
                    axes_order.push_back(spatial_dims + 1);
                    for (size_t i = 2; i < data_shape.size(); ++i)
                    {
                        axes_order.push_back(spatial_dims + i);
                        axes_order.push_back(i - 1);
                    }
                    break;
                }
                }
                std::vector<size_t> plain_axes_order(data_shape.size());
                std::iota(plain_axes_order.begin(), plain_axes_order.end(), 0);
                std::vector<char> dispersed_data(shape_size(data_shape) * elem_size);
                std::vector<char> transposed_data(shape_size(data_shape) * elem_size);

                runtime::opt_kernel::reshape(data,
                                             dispersed_data.data(),
                                             data_shape,
                                             plain_axes_order,
                                             dispersed_shape,
                                             elem_size);

                Shape post_transpose_shape(axes_order.size());
                for (size_t axis_idx = 0; axis_idx < axes_order.size(); ++axis_idx)
                {
                    post_transpose_shape[axis_idx] = dispersed_shape[axes_order[axis_idx]];
                }
                runtime::opt_kernel::reshape(dispersed_data.data(),
                                             transposed_data.data(),
                                             dispersed_shape,
                                             axes_order,
                                             post_transpose_shape,
                                             elem_size);

                Shape squeezed_shape{n_dim, c_flat};
                for (size_t i = spatial_dim_index; i < data_shape.size(); ++i)
                {
                    squeezed_shape.push_back(data_shape.at(i) * block_size);
                }
                for (size_t i = plain_axes_order.size() - 1; i < post_transpose_shape.size() - 1;
                     ++i)
                {
                    plain_axes_order.push_back(plain_axes_order[i] + 1);
                }
                runtime::opt_kernel::reshape(transposed_data.data(),
                                             out,
                                             post_transpose_shape,
                                             plain_axes_order,
                                             squeezed_shape,
                                             elem_size);
                return true;
            }

        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
