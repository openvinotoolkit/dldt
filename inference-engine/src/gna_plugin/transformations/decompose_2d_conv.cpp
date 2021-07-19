// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/decompose_2d_conv.hpp"

#include <memory>

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pass/manager.hpp>
#include <ie_common.h>
#include "backend/gna_limitations.hpp"


using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(Decompose2DConv, "Decompose2DConv", 0);
NGRAPH_RTTI_DEFINITION(Decompose2DConvTransposedWithBias, "Decompose2DConvTransposedWithBias", 0);
NGRAPH_RTTI_DEFINITION(Decompose2DConvTransposedWithBiasAF, "Decompose2DConvTransposedWithBiasAF", 0);

#define GNA_MAX_PERMUTE_COL_COUNT 8


struct GraphData {
    std::shared_ptr<ngraph::opset7::Transpose>leading_transpose;
    std::shared_ptr<ngraph::opset7::Convolution>conv;
    std::shared_ptr<ngraph::opset7::Transpose>trailing_transpose;
    std::shared_ptr<ngraph::op::util::UnaryElementwiseArithmetic>af;
    std::shared_ptr<ngraph::opset7::MaxPool>max_pool;
    std::shared_ptr<ngraph::Node>last_op_in_sequence_for_replacement;
    std::shared_ptr<ngraph::Node>bias_const;
    size_t conv_count;
};

struct ConvData {
    size_t input_height;
    size_t input_width;
    size_t input_channel_count;
    size_t filter_height;
    size_t filter_width;
    size_t filter_count;
    size_t filter_channel_count;
    size_t filter_dilation_height;
    size_t filter_dilation_width;
    size_t filter_stride_height;
    size_t filter_stride_width;
    size_t pads_begin_height;
    size_t pads_begin_width;
    size_t pads_end_height;
    size_t pads_end_width;
    size_t output_height;
    size_t output_width;
    ngraph::op::PadType padding_type;
    size_t output_channel_count;
    ngraph::element::Type element_type;
};

struct MaxPoolData {
    size_t pool_size_width;
    size_t pool_stride_width;
    // TODO: currently 2D max pool is not supported
    //size_t pool_size_height;
    //size_t pool_stride_height;
};

static bool VerifyAndGetConvParams(std::shared_ptr<ngraph::opset7::Convolution> conv, ConvData& conv_data) {
    const auto& input = conv->input_value(0);
    const auto& filters = conv->input_value(1);

    // We support only 2D conv batch 1
    if (conv->get_dilations().size() != 2 ||
        conv->get_strides().size() != 2 ||
        input.get_shape()[0] != 1) {
        return false;
    }

    size_t filter_height = filters.get_shape()[2];
    size_t filter_width = filters.get_shape()[3];

    if (filter_width > GNA_MAX_PERMUTE_COL_COUNT || filter_height > GNA_MAX_PERMUTE_COL_COUNT) {
        return false;
    }

    conv_data.output_height = conv->get_output_shape(0)[2];
    conv_data.output_width = conv->get_output_shape(0)[3];
    conv_data.padding_type = conv->get_auto_pad();
    conv_data.input_channel_count = conv->input_value(0).get_shape()[1];
    conv_data.input_height = conv->input_value(0).get_shape()[2];
    conv_data.input_width = conv->input_value(0).get_shape()[3];
    conv_data.filter_count = conv->input_value(1).get_shape()[0];
    conv_data.filter_channel_count = conv->input_value(1).get_shape()[1];
    conv_data.filter_height = conv->input_value(1).get_shape()[2];
    conv_data.filter_width = conv->input_value(1).get_shape()[3];
    conv_data.filter_dilation_height = conv->get_dilations()[0];
    conv_data.filter_dilation_width = conv->get_dilations()[1];
    conv_data.filter_stride_height = conv->get_strides()[0];
    conv_data.filter_stride_width = conv->get_strides()[1];
    conv_data.pads_begin_height = conv->get_pads_begin()[0];
    conv_data.pads_begin_width = conv->get_pads_begin()[1];
    conv_data.pads_end_height = conv->get_pads_end()[0];
    conv_data.pads_end_width = conv->get_pads_end()[1];
    conv_data.output_channel_count = conv_data.filter_count;
    conv_data.element_type = conv->get_element_type();

    IE_ASSERT(conv_data.output_channel_count == conv->get_output_shape(0)[1]);

    return true;
}

static bool TransposeOrderMatches(std::shared_ptr<ngraph::opset7::Transpose> transpose, std::vector<int64_t> order) {
    if (!transpose)
        return false;
    const ngraph::Output<ngraph::Node>& transpose_order = transpose->input_value(1);
    auto transpose_order_dim = transpose_order.get_shape().size();

    if (transpose_order_dim != 1 || transpose_order.get_shape()[0] != order.size())
        return false;

    auto const_with_order_values = std::dynamic_pointer_cast<ngraph::opset7::Constant>(transpose_order.get_node_shared_ptr());
    if (!const_with_order_values)
        return false;

    const int64_t* data = const_with_order_values->get_data_ptr<int64_t>();
    if (!data)
        return false;

    for (size_t i = 0; i < order.size(); i++) {
        if (order[i] != data[i])
            return false;
    }

    return true;
}

template<typename T>
static std::shared_ptr<ngraph::Node> VerifyBiasAndCreateConst(std::shared_ptr<ngraph::opset7::Add> conv_bias, const ConvData& conv_data) {
    auto add_const = std::dynamic_pointer_cast<ngraph::op::Constant>(conv_bias->input_value(1).get_node_shared_ptr());

    if (add_const) {
        auto bias_size = shape_size(add_const->get_shape());

        // The add may be a normal add not conv bias, then we just go further
        if (bias_size == conv_data.filter_count) {
            const auto* srd_data_pointer = add_const->get_data_ptr<T>();
            std::vector<T> bias_values(srd_data_pointer, srd_data_pointer + bias_size);
            return ngraph::opset7::Constant::create(conv_data.element_type, ngraph::Shape{1, bias_size , 1, 1}, bias_values);
        }
    }
    // Bias size does not match (or dynamic bias), can't decompose such convolution
    return nullptr;
}

static bool VerifyMaxPool(std::shared_ptr<ngraph::opset7::MaxPool> max_pool, MaxPoolData& pool_data) {
    // Check if MaxPool vertical stride == pool size
    auto pool_strides = max_pool->get_strides();
    auto pool_filter = max_pool->get_kernel();

    // Check Max Pool padding and limitations
    if (max_pool->get_auto_pad() != ngraph::op::PadType::VALID ||
        pool_filter.size() != 2 || pool_strides.size() != 2 ||
        pool_filter[0] > GNALimitations::maxPoolMaxWindowSize)
        return false;

    pool_data.pool_size_width = pool_filter[1];
    pool_data.pool_stride_width = pool_strides[1];
    return true;
}

static bool ShouldDecompose(GraphData& graph_data, const ConvData& conv_data, const MaxPoolData& maxpool_data) {
    // Check if split of plane due to GNA HW limitations of 768 filter elements is possible
    graph_data.conv_count = 1;
    size_t total_factorized_conv_channel_count = (conv_data.input_channel_count * conv_data.filter_height * conv_data.filter_width);
    while (total_factorized_conv_channel_count / graph_data.conv_count > GNALimitations::convFilterMaxSize ||
        total_factorized_conv_channel_count % graph_data.conv_count != 0 || conv_data.filter_channel_count % graph_data.conv_count != 0)
        graph_data.conv_count++;

    // Currently we are able to split only convolutions without pooling in horizontal dimension
    if (graph_data.conv_count > GNA_MAX_PERMUTE_COL_COUNT ||
        ((maxpool_data.pool_size_width > 1 || maxpool_data.pool_stride_width > 1) && graph_data.conv_count > 1))
        return false;

    // GNA supported features - there is no need to decompose such convolution
    if (graph_data.conv_count == 1 && (conv_data.input_height == 1 || conv_data.input_width == 1) &&
        conv_data.filter_dilation_width == 1 && conv_data.filter_dilation_height == 1)
        return false;

    return true;
}

static std::shared_ptr<ngraph::opset7::StridedSlice> FlatCrop(ngraph::Output<ngraph::Node> input, size_t offset, size_t size) {
    auto shape = input.get_shape();
    return std::make_shared<ngraph::opset7::StridedSlice>(
        input, // data
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {(size_t)0, offset}), // begin slice index
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {(size_t)0, offset + size}), // end slice index
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {(size_t)1, (size_t)1}), // strides
        std::vector<int64_t>{1, 0},  // begin mask
        std::vector<int64_t>{1, 0}); // end mask
}

template<typename T>
static std::vector<std::shared_ptr<ngraph::opset7::Constant>> Split2DConvFilters(std::shared_ptr<ngraph::opset7::Constant>& filters,
    const bool& vertical_permute, const bool& horizontal_permute, const ngraph::element::Type& element_type, const size_t& split_channels) {

    std::vector <std::shared_ptr<ngraph::opset7::Constant>> result;
    auto filter_shape = filters->get_output_shape(0);
    if (!horizontal_permute && !vertical_permute && split_channels == 1)
        return {filters};

    IE_ASSERT(filter_shape.size() == 4);

    std::vector<std::vector<float>> flat_filters;
    flat_filters.resize(split_channels);
    for (size_t i = 0; i < split_channels; i++)
        flat_filters[i].resize(shape_size(filter_shape) / split_channels);

    auto N = filter_shape[0];
    auto C = filter_shape[1];
    auto H = filter_shape[2];
    auto W = filter_shape[3];

    size_t CS = (C / split_channels);
    const auto* data = filters->get_data_ptr<T>();

    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < CS; c++) {
            for (size_t s = 0; s < split_channels; s++) {
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        if (vertical_permute || !horizontal_permute) {
                            flat_filters[s][n * CS * H * W + c * H * W + h * W + w] =
                                data[n * C * H * W + (c * split_channels + s) * H * W + h * W + w];
                        } else if (horizontal_permute) {
                            flat_filters[s][n * CS * H * W + c * H * W + H * w + h] =
                                data[n * C * H * W + (c * split_channels + s) * H * W + h * W + w];
                        }
                    }
                }
            }
        }
    }

    if (vertical_permute && horizontal_permute) {
        for (auto new_filter : flat_filters)
            result.push_back(std::make_shared<ngraph::opset7::Constant>(element_type,
                ngraph::Shape{filter_shape[0], filter_shape[1] * filter_shape[2] * filter_shape[3] / split_channels, 1, 1}, new_filter));
    } else if (vertical_permute && !horizontal_permute) {
        for (auto new_filter : flat_filters)
            result.push_back(std::make_shared<ngraph::opset7::Constant>(element_type,
                ngraph::Shape{filter_shape[0], filter_shape[1] * filter_shape[2] / split_channels, 1, filter_shape[3]}, new_filter));
    } else if (!vertical_permute && horizontal_permute) {
        for (auto new_filter : flat_filters)
            result.push_back(std::make_shared<ngraph::opset7::Constant>(element_type,
                ngraph::Shape{filter_shape[0], filter_shape[1] * filter_shape[3] / split_channels, filter_shape[2], 1}, new_filter));
    } else {
        for (auto new_filter : flat_filters)
            result.push_back(std::make_shared<ngraph::opset7::Constant>(element_type,
                ngraph::Shape{filter_shape[0], filter_shape[1] / split_channels, filter_shape[2], filter_shape[3]}, new_filter));
    }

    return result;
}

static std::vector<std::shared_ptr<ngraph::opset7::Constant>> CreateSplit(const GraphData& graph_data,
    ConvData& conv_data, ngraph::OutputVector& split_planes) {
    const ngraph::Output<ngraph::Node>& filters = graph_data.conv->input_value(1);
    auto filter_values = std::dynamic_pointer_cast<ngraph::opset7::Constant>(filters.get_node_shared_ptr());

    // We need to have proper input shape first
    auto padded_input_plane = std::make_shared<ngraph::opset7::Reshape>(graph_data.leading_transpose->input_value(0),
        ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{2},
            ngraph::Shape{1ull, shape_size(graph_data.leading_transpose->input_value(0).get_shape())}), false);
    copy_runtime_info(graph_data.conv, padded_input_plane);

    if (graph_data.conv_count > 1) {
        auto reshape_before_transpose = std::make_shared<ngraph::opset7::Reshape>(padded_input_plane,
            ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{2},
                {shape_size(padded_input_plane->get_shape()) / graph_data.conv_count, graph_data.conv_count}), false);

        auto transpose_before_channel_wise_split = std::make_shared<ngraph::opset7::Transpose>(reshape_before_transpose,
            ngraph::op::Constant::create(ngraph::element::Type_t::i64, ngraph::Shape{2}, {1ll, 0ll})->output(0));

        const auto axis_node = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        const auto split = std::make_shared<ngraph::opset7::Split>(transpose_before_channel_wise_split, axis_node, graph_data.conv_count);
        split_planes = split->outputs();
    } else {
        split_planes.push_back(padded_input_plane);
    }

    // If the input plane exceeds GNA limits and we have split into several convolutions, then we need to split filter/kernel data as well;
    // we also need to take filter height and potential dilation into account when modifying the filters
    bool vertical_permute = (conv_data.filter_height > 1);
    bool horizontal_permute = (conv_data.filter_dilation_width > 1);

    std::vector<std::shared_ptr<ngraph::opset7::Constant>> h_1_filters{};

    if (conv_data.element_type == ngraph::element::f32) {
        h_1_filters = Split2DConvFilters<float>(filter_values, vertical_permute, horizontal_permute, conv_data.element_type, graph_data.conv_count);
    } else {
        h_1_filters = Split2DConvFilters<ngraph::float16>(filter_values, vertical_permute, horizontal_permute, conv_data.element_type, graph_data.conv_count);
    }

    for (auto filter : h_1_filters)
        copy_runtime_info(graph_data.conv, filter);

    // If we have split input plane and convolutions due to GNA limitation - we must sum their results at the end
    conv_data.input_channel_count /= graph_data.conv_count;

    return h_1_filters;
}

static void TransformInput(const GraphData& graph_data, const ConvData& conv_data, ngraph::Output<ngraph::Node>& split_input_plane) {
    /*
    *              Padded row - NHWC order
    *                  |
    *        Split in vertical dim (filter height)
    *                / | \
    *                Concat
    *                  |
    *              Transpose
    */

    // First we need to prepare flat (height = 1) slices of input data proper for flattened (height = 1) filter size
    ngraph::OutputVector dilated_input_planes;
    for (size_t filter_height = 0; filter_height < conv_data.filter_height; filter_height++) {
        size_t offset = filter_height * conv_data.filter_dilation_height * (conv_data.pads_begin_width + conv_data.input_width + conv_data.pads_end_width) *
            conv_data.input_channel_count;
        auto slice = FlatCrop(split_input_plane, offset, (conv_data.pads_begin_width + conv_data.input_width + conv_data.pads_end_width) *
            conv_data.input_channel_count * conv_data.output_height);
        copy_runtime_info(graph_data.conv, slice);
        dilated_input_planes.push_back(slice);
    }

    // Interleaving dilated input planes
    auto dilated_chunks_concat = std::make_shared<ngraph::opset7::Concat>(dilated_input_planes, 0);

    auto transposed_dilated_chunks = std::make_shared<ngraph::op::Transpose>(dilated_chunks_concat,
        ngraph::op::Constant::create(ngraph::element::Type_t::i64, ngraph::Shape{2}, {1ll, 0ll})->output(0));

    // Flattening of interleaved input planes
    auto flattened_dilated_transposed_input = std::make_shared<ngraph::opset7::Reshape>(transposed_dilated_chunks,
        ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{2},
            {(size_t)1, (conv_data.pads_begin_width + conv_data.input_width + conv_data.pads_end_width) *
        conv_data.input_channel_count * conv_data.output_height * conv_data.filter_height}), false);

    copy_runtime_info(graph_data.conv, {dilated_chunks_concat, flattened_dilated_transposed_input, transposed_dilated_chunks });
    split_input_plane = flattened_dilated_transposed_input;
}

static std::shared_ptr<ngraph::Node> GenerateDeomposedConv(const GraphData& graph_data, ConvData& conv_data,
    const MaxPoolData& pool_data, ngraph::Output<ngraph::Node>& reduced_input_plane,
    const std::vector<std::shared_ptr<ngraph::opset7::Constant>>& h_1_filters, const size_t conv_index) {
    ngraph::OutputVector result_chunks;
    std::shared_ptr<ngraph::Node> last_op;
    bool horizontal_permute = (conv_data.filter_dilation_width > 1);
    size_t h_1_filter_channel_count = (conv_data.input_channel_count * conv_data.filter_height);
    size_t padded_row_width = conv_data.pads_begin_width + conv_data.input_width + conv_data.pads_end_width;

    for (size_t output_height = 0; output_height < conv_data.output_height; output_height += conv_data.filter_stride_height) {
        size_t offset = output_height * padded_row_width * h_1_filter_channel_count;
        auto row = (conv_data.output_height == 1) ? reduced_input_plane :
            FlatCrop(reduced_input_plane, offset, padded_row_width * h_1_filter_channel_count);
        /*
            *              Padded row
            *                  |
            *        ??? <Dilation !=1> ???
            *                  |
            *         Split in vertical dim
            *                / | \
            *                Concat
            *                  |
            *               Permute
            *                  |
            *              Transpose (NHWC => NCHW)
            *                  |
            *                1D Conv (Bias | MaxPooling)
            *                  |
            *              Transpose (NCHW => NHWC)
            */
        auto nhwc_conv_y_input = row;

        // Valid 1D (decomposed 2D) convolution wrapped with transposes NHWC => NCHW => conv => NCHW => NHWC
        auto nhwc_conv_1d = [](std::shared_ptr<ngraph::Node> source_conv2d,
            ngraph::Output<ngraph::Node> input,
            std::shared_ptr<ngraph::Node> filters,
            std::shared_ptr<ngraph::Node> add_bias_const,
            size_t stride_width,
            size_t pool_size_width,
            size_t pool_stride_width,
            ngraph::op::RoundingType rounding_type,
            std::shared_ptr<ngraph::op::util::UnaryElementwiseArithmetic> af,
            size_t h_index,
            size_t c_index = 0) {
                // Transpose NHWC => NCHW
                std::shared_ptr<ngraph::Node> nchw_input = std::make_shared<ngraph::op::Transpose>(input,
                    ngraph::op::Constant::create(ngraph::element::Type_t::i64, ngraph::Shape{4}, {0ll, 3ll, 1ll, 2ll})->output(0));

                // 1D Convolution
                auto conv = std::make_shared<ngraph::opset7::Convolution>(nchw_input, filters,
                    ngraph::Strides{1, stride_width}, ngraph::CoordinateDiff{0, 0}, ngraph::CoordinateDiff{0, 0},
                    ngraph::Strides{1, 1}, ngraph::op::PadType::VALID);
                std::string conv_name = source_conv2d->get_friendly_name() + "_H_" + std::to_string(h_index) + "_CH_" + std::to_string(c_index);
                conv->set_friendly_name(conv_name);

                // Bias
                std::shared_ptr<ngraph::Node> last_conv_block_op = conv;
                if (add_bias_const) {
                    last_conv_block_op = std::make_shared<ngraph::opset7::Add>(conv, add_bias_const);
                    copy_runtime_info(source_conv2d, last_conv_block_op);
                }

                // Max pooling
                if (pool_size_width > 1 || pool_stride_width > 1) {
                    last_conv_block_op = std::make_shared<ngraph::opset7::MaxPool>(last_conv_block_op, ngraph::Strides{1, pool_stride_width},
                        ngraph::Shape{0, 0}, ngraph::Shape{0, 0}, ngraph::Shape{1, pool_size_width}, rounding_type, ngraph::op::PadType::VALID);
                }

                // Activation function
                if (af) {
                    auto af_result = af->copy_with_new_inputs({last_conv_block_op});
                    copy_runtime_info(conv, af_result);
                    last_conv_block_op = af_result;
                }

                // Transpose NCHW => NHWC
                auto nhwc_output = std::make_shared<ngraph::op::Transpose>(last_conv_block_op,
                    ngraph::op::Constant::create(ngraph::element::Type_t::i64, ngraph::Shape{4}, {0ull, 2ull, 3ull, 1ull})->output(0));
                copy_runtime_info(source_conv2d, {nchw_input, conv, nhwc_output});
                return nhwc_output;
        };

        if (horizontal_permute) {
            // Horizontal split - transform input accordingly
            ngraph::OutputVector dilated_chunks;
            for (size_t filter_width = 0; filter_width < conv_data.filter_width; filter_width++) {
                size_t offset = filter_width * conv_data.filter_dilation_width * h_1_filter_channel_count;
                auto slice = FlatCrop(row, offset, h_1_filter_channel_count * conv_data.output_width);
                copy_runtime_info(graph_data.conv, slice);
                dilated_chunks.push_back(slice);
            }

            // Concat
            auto dilated_chunks_concat = std::make_shared<ngraph::opset7::Concat>(dilated_chunks, 0);

            // Transpose
            auto transposed_dilated_chunks = std::make_shared<ngraph::op::Transpose>(dilated_chunks_concat,
                ngraph::op::Constant::create(ngraph::element::Type_t::i64, ngraph::Shape{2}, {1ll, 0ll})->output(0));

            // Flatten
            auto flatten_dilated_conv_input = std::make_shared<ngraph::opset7::Reshape>(transposed_dilated_chunks,
                ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{4},
                    ngraph::Shape{1ull, 1ull, conv_data.output_width, h_1_filter_channel_count * conv_data.filter_width}), false);

            copy_runtime_info(graph_data.conv, ngraph::NodeVector{flatten_dilated_conv_input, transposed_dilated_chunks, dilated_chunks_concat});

            nhwc_conv_y_input = flatten_dilated_conv_input;
        } else {
            // If no horizontal split is done, only reshape is required before decomposed convolution
            nhwc_conv_y_input = std::make_shared<ngraph::opset7::Reshape>(nhwc_conv_y_input,
                ngraph::op::Constant::create(ngraph::element::Type_t::i64, ngraph::Shape{4},
                    ngraph::Shape{1ull, 1ull, padded_row_width, h_1_filter_channel_count}), false);
        }

        // Pointwise convolutions
        // Valid 1D convolution wrapped with transposes NHWC => NCHW => Conv => NCHW => NHWC
        // Activation function can be fused with convolution only if it is not split
        auto nhwc_y_output = nhwc_conv_1d(graph_data.conv, nhwc_conv_y_input, h_1_filters[conv_index], conv_index ? nullptr : graph_data.bias_const,
            conv_data.filter_stride_width, pool_data.pool_size_width, pool_data.pool_stride_width,
            graph_data.max_pool ? graph_data.max_pool->get_rounding_type() : ngraph::op::RoundingType::FLOOR,
            graph_data.conv_count == 1 ? graph_data.af : nullptr, output_height);
        result_chunks.push_back(nhwc_y_output);
        last_op = nhwc_y_output;
    }

    // H dimemsion greater than 1
    if (result_chunks.size() > 1) {
        // Concat in H dim
        // In NHWC index of H is 1
        auto concatenated_sub_results = std::make_shared<ngraph::opset7::Concat>(result_chunks, 1);
        copy_runtime_info(graph_data.conv, concatenated_sub_results);
        last_op = concatenated_sub_results;
    }
    return last_op;
}

static void Decompose(const GraphData& graph_data, ConvData& conv_data, const MaxPoolData& pool_data) {
    ngraph::OutputVector split_planes;
    std::vector<std::shared_ptr<ngraph::Node>> partial_conv_results;

    // Split input and filters due to GNA HW limitations
    auto h_1_filters = CreateSplit(graph_data, conv_data, split_planes);

    // Do transformations in each of the splits created above (GNA filter channel count limit)
    for (size_t conv_index = 0; conv_index < graph_data.conv_count; conv_index++) {
        ngraph::Output<ngraph::Node>& split_input_plane = split_planes[conv_index];

        // Input data needs to be prepared before 2D convolution decomposition
        if (conv_data.filter_height > 1) {
            TransformInput(graph_data, conv_data, split_input_plane);
        }

        auto flat_conv = GenerateDeomposedConv(graph_data, conv_data, pool_data, split_input_plane, h_1_filters, conv_index);
        partial_conv_results.push_back(flat_conv);
    }

    std::shared_ptr<ngraph::Node> conv_result = partial_conv_results.front();
    for (size_t i = 1; i < partial_conv_results.size(); i++) {
        auto add_result = std::make_shared<ngraph::opset7::Add>(partial_conv_results[i], conv_result);
        copy_runtime_info(graph_data.conv, add_result);
        conv_result = add_result;
    }

    // TODO: Max Pool 2D case
    //if (graph_data.max_pool && (maxpool_data.pool_size_height > 1 || maxpool_data.pool_stride_height > 1)) {
    //}

    // TODO: Check if bias after transpose should not be added after summing all convolutions instead of just replacing
    // Activation function after trailing Transpose NCHW->NHWC
    if (graph_data.af && graph_data.conv_count > 1) {
        auto af_result = graph_data.af->copy_with_new_inputs({conv_result});
        copy_runtime_info(graph_data.conv, af_result);
        conv_result = af_result;
    }
    // We need to put the same name as before for the Convolution layer, so its output can be used as network result
    std::string conv_result_name = graph_data.last_op_in_sequence_for_replacement->get_friendly_name();
    replace_node(graph_data.last_op_in_sequence_for_replacement, conv_result);
    conv_result->set_friendly_name(conv_result_name);
}

static bool Convert(std::shared_ptr<ngraph::Node> leading_transpose,
    std::shared_ptr<ngraph::Node> conv,
    std::shared_ptr<ngraph::Node> trailing_transpose,
    std::shared_ptr<ngraph::Node> bias,
    std::shared_ptr<ngraph::Node> max_pool,
    std::shared_ptr<ngraph::Node> af,
    std::shared_ptr<ngraph::Node> last_op_for_replacement) {

    GraphData graph_data{std::dynamic_pointer_cast<ngraph::opset7::Transpose>(leading_transpose),
        std::dynamic_pointer_cast<ngraph::opset7::Convolution>(conv),
        std::dynamic_pointer_cast<ngraph::opset7::Transpose>(trailing_transpose),
        std::dynamic_pointer_cast<ngraph::op::util::UnaryElementwiseArithmetic>(af),
        std::dynamic_pointer_cast<ngraph::opset7::MaxPool>(max_pool),
        last_op_for_replacement};
    ConvData conv_data;
    MaxPoolData maxpool_data{1, 1};

    if (!VerifyAndGetConvParams(std::dynamic_pointer_cast<ngraph::opset7::Convolution>(conv), conv_data))
        return false;

    // We are looking for Transpose(NHWC->NCHW) => Conv => Transpose(NCHW->NHWC)
    // or similar cases, so required network must be in NHWC order like in TF
    if (!TransposeOrderMatches(std::dynamic_pointer_cast<ngraph::opset7::Transpose>(leading_transpose), {0, 3, 1, 2}))
        return false;

    if (!TransposeOrderMatches(std::dynamic_pointer_cast<ngraph::opset7::Transpose>(trailing_transpose), {0, 2, 3, 1}))
        return false;

    if (conv_data.element_type == ngraph::element::f32) {
        if (bias && !(graph_data.bias_const = VerifyBiasAndCreateConst<float>(std::dynamic_pointer_cast<ngraph::opset7::Add>(bias), conv_data)))
            return false;
    } else {
        if (bias && !(graph_data.bias_const = VerifyBiasAndCreateConst<ngraph::float16>(std::dynamic_pointer_cast<ngraph::opset7::Add>(bias), conv_data)))
            return false;
    }

    if (max_pool && !VerifyMaxPool(std::dynamic_pointer_cast<ngraph::opset7::MaxPool>(max_pool), maxpool_data))
        return false;

    if (!ShouldDecompose(graph_data, conv_data, maxpool_data))
        return false;

    // All checks applied - now we may start decomposition
    Decompose(graph_data, conv_data, maxpool_data);

    return true;
}

static std::function<bool(ngraph::Output<ngraph::Node>)> consumers_and_rank(const size_t expected_count, const ngraph::Dimension& expected_rank) {
    return [=](ngraph::Output<ngraph::Node> output) -> bool {
        return ngraph::pattern::consumers_count(expected_count) && ngraph::pattern::rank_equals(expected_rank);
    };
}

static bool VerifyBias(std::shared_ptr<ngraph::Node> conv, std::shared_ptr<ngraph::Node> bias) {
    auto add_const = std::dynamic_pointer_cast<ngraph::op::Constant>(bias->input_value(1).get_node_shared_ptr());

    if (!add_const) {
        add_const = std::dynamic_pointer_cast<ngraph::op::Constant>(bias->input_value(0).get_node_shared_ptr());
    }

    if (!add_const) {
        auto bias_size = shape_size(add_const->get_shape());
        auto conv_filter_count = conv->input_value(1).get_shape()[0];
        if (bias_size == conv_filter_count)
            return true;
    }
    return false;
}

Decompose2DConv::Decompose2DConv() {
    MATCHER_SCOPE(Decompose2DConv);

    auto const_input = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto leading_transpose = ngraph::pattern::wrap_type<ngraph::opset7::Transpose>({ngraph::pattern::any_input(), const_input},
        consumers_and_rank(1, 4));
    auto conv = ngraph::pattern::wrap_type<ngraph::opset7::Convolution>(
        {leading_transpose, ngraph::pattern::wrap_type<ngraph::opset7::Constant>(ngraph::pattern::rank_equals(4))},
        ngraph::pattern::consumers_count(1));
    auto bias = ngraph::pattern::wrap_type<ngraph::opset7::Add>({conv, const_input},
        ngraph::pattern::consumers_count(1));
    auto max_pool = ngraph::pattern::wrap_type<ngraph::opset7::MaxPool>({bias},
        ngraph::pattern::consumers_count(1));
    auto af1 = ngraph::pattern::wrap_type<ngraph::opset7::Relu, ngraph::opset7::Sigmoid,
        ngraph::opset7::Tanh, ngraph::opset7::Abs, ngraph::opset7::Log, ngraph::opset7::Exp,
        ngraph::opset7::Sign, ngraph::opset7::Clamp>({bias}, ngraph::pattern::consumers_count(1));
    auto af2 = ngraph::pattern::wrap_type<ngraph::opset7::Relu, ngraph::opset7::Sigmoid,
        ngraph::opset7::Tanh, ngraph::opset7::Abs, ngraph::opset7::Log, ngraph::opset7::Exp,
        ngraph::opset7::Sign, ngraph::opset7::Clamp>({max_pool}, ngraph::pattern::consumers_count(1));
    auto transpose_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{conv, bias, max_pool, af1, af2});
    auto trailing_transpose = ngraph::pattern::wrap_type<ngraph::opset7::Transpose>({transpose_input, const_input},
        consumers_and_rank(1, 4));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto bias_it = pattern_map.find(bias);
        auto bias_node = (bias_it == std::end(pattern_map) ? nullptr : bias_it->second.get_node_shared_ptr());
        auto max_pool_it = pattern_map.find(max_pool);
        auto max_pool_node = (max_pool_it == std::end(pattern_map) ? nullptr : max_pool_it->second.get_node_shared_ptr());
        auto af1_it = pattern_map.find(af1);
        auto af2_it = pattern_map.find(af2);
        auto af_node = (af1_it == std::end(pattern_map) ?
            ((af2_it == std::end(pattern_map) ? nullptr : af2_it->second.get_node_shared_ptr())) : af1_it->second.get_node_shared_ptr());

        return Convert(pattern_map.at(leading_transpose).get_node_shared_ptr(), pattern_map.at(conv).get_node_shared_ptr(),
            pattern_map.at(trailing_transpose).get_node_shared_ptr(), bias_node, max_pool_node, af_node,
            pattern_map.at(trailing_transpose).get_node_shared_ptr());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(trailing_transpose, matcher_name);
    this->register_matcher(m, callback);
}

Decompose2DConvTransposedWithBias::Decompose2DConvTransposedWithBias() {
    MATCHER_SCOPE(Decompose2DConvTransposedWithBias);

    auto const_input_i64 = ngraph::pattern::wrap_type<ngraph::opset7::Constant>(ngraph::pattern::type_matches(ngraph::element::i64));
    auto const_input = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto leading_transpose = ngraph::pattern::wrap_type<ngraph::opset7::Transpose>({ngraph::pattern::any_input(), const_input_i64},
        consumers_and_rank(1, 4));
    auto conv = ngraph::pattern::wrap_type<ngraph::opset7::Convolution>(
        {leading_transpose, ngraph::pattern::wrap_type<ngraph::opset7::Constant>(ngraph::pattern::rank_equals(4))},
        ngraph::pattern::consumers_count(1));
    auto trailing_transpose = ngraph::pattern::wrap_type<ngraph::opset7::Transpose>({conv, const_input_i64},
        consumers_and_rank(1, 4));
    auto bias = ngraph::pattern::wrap_type<ngraph::opset7::Add>({trailing_transpose, const_input},
        ngraph::pattern::consumers_count(1));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        if (!VerifyBias(pattern_map.at(conv).get_node_shared_ptr(), pattern_map.at(bias).get_node_shared_ptr()))
            return false;

        return Convert(pattern_map.at(leading_transpose).get_node_shared_ptr(), pattern_map.at(conv).get_node_shared_ptr(),
            pattern_map.at(trailing_transpose).get_node_shared_ptr(), pattern_map.at(bias).get_node_shared_ptr(), nullptr, nullptr,
            pattern_map.at(bias).get_node_shared_ptr());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(bias, matcher_name);
    this->register_matcher(m, callback);
}

Decompose2DConvTransposedWithBiasAF::Decompose2DConvTransposedWithBiasAF() {
    MATCHER_SCOPE(Decompose2DConvTransposedWithBiasAF);

    auto const_input_i64 = ngraph::pattern::wrap_type<ngraph::opset7::Constant>(ngraph::pattern::type_matches(ngraph::element::i64));
    auto const_input = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto leading_transpose = ngraph::pattern::wrap_type<ngraph::opset7::Transpose>({ngraph::pattern::any_input(), const_input_i64},
        consumers_and_rank(1, 4));
    auto conv = ngraph::pattern::wrap_type<ngraph::opset7::Convolution>(
        {leading_transpose, ngraph::pattern::wrap_type<ngraph::opset7::Constant>(ngraph::pattern::rank_equals(4))},
        ngraph::pattern::consumers_count(1));
    auto trailing_transpose = ngraph::pattern::wrap_type<ngraph::opset7::Transpose>({conv, const_input_i64},
        consumers_and_rank(1, 4));
    auto bias = ngraph::pattern::wrap_type<ngraph::opset7::Add>({trailing_transpose, const_input},
        ngraph::pattern::consumers_count(1));
    auto af = ngraph::pattern::wrap_type<ngraph::opset7::Relu, ngraph::opset7::Sigmoid,
        ngraph::opset7::Tanh, ngraph::opset7::Abs, ngraph::opset7::Log, ngraph::opset7::Exp,
        ngraph::opset7::Sign, ngraph::opset7::Clamp>({bias},
        ngraph::pattern::consumers_count(1));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        if (!VerifyBias(pattern_map.at(conv).get_node_shared_ptr(), pattern_map.at(bias).get_node_shared_ptr()))
            return false;

        return Convert(pattern_map.at(leading_transpose).get_node_shared_ptr(), pattern_map.at(conv).get_node_shared_ptr(),
            pattern_map.at(trailing_transpose).get_node_shared_ptr(), pattern_map.at(bias).get_node_shared_ptr(),
            nullptr, pattern_map.at(af).get_node_shared_ptr(), pattern_map.at(af).get_node_shared_ptr());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(af, matcher_name);
    this->register_matcher(m, callback);
}
