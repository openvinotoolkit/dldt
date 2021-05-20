// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <string>
#include <vector>
#include <algorithm>

#include <ngraph/opsets/opset3.hpp>
#include "ie_parallel.hpp"
#include "mkldnn_bucketize_node.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNBucketizeNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto bucketsize = std::dynamic_pointer_cast<const ngraph::opset3::Bucketize>(op);
        if (!bucketsize) {
            errorMessage = "Only opset3 Bucketize operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNBucketizeNode::MKLDNNBucketizeNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
                                     MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "Bucketize layer with name '" + op->get_friendly_name() + "' ";
    const auto bucketsize = std::dynamic_pointer_cast<const ngraph::opset3::Bucketize>(op);

    if (op->get_input_size() != 2 || op->get_output_size() != 1) {
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";
    }

    // check one attribute
    with_right = bucketsize->get_with_right_bound();

    // check precisions for input and output tensors
    input_precision = details::convertPrecision(op->get_input_element_type(INPUT_TENSOR_PORT));
    if (input_precision != Precision::FP32 && input_precision != Precision::I32 &&
        input_precision != Precision::I64) {
        input_precision = Precision::FP32;
    }
    boundaries_precision = details::convertPrecision(op->get_input_element_type(INPUT_BINS_PORT));
    if (boundaries_precision != Precision::FP32 && boundaries_precision != Precision::I32 &&
        boundaries_precision != Precision::I64) {
        boundaries_precision = Precision::FP32;
    }
    output_precision = details::convertPrecision(op->get_output_element_type(OUTPUT_TENSOR_PORT));
    if (output_precision != Precision::I32 && output_precision != Precision::I64) {
        output_precision = Precision::I32;
    }

    // check dimensions of input tensors
    SizeVector input_tensor_dims = op->get_input_shape(INPUT_TENSOR_PORT);
    if (input_tensor_dims.size() < 1) {
        IE_THROW() << errorPrefix << " has incorrect dimensions of the input.";
    }
    SizeVector input_bin_dims = op->get_input_shape(INPUT_BINS_PORT);
    if (input_bin_dims.size() != 1) {
        IE_THROW() << errorPrefix << " has incorrect dimensions of the boundaries tensor.";
    }
    if (input_bin_dims[0] != 0) {
        with_bins = true;
    }
    num_bin_values = input_bin_dims[0];

    num_values = std::accumulate(input_tensor_dims.begin(), input_tensor_dims.end(), size_t(1), std::multiplies<size_t>());
}

void MKLDNNBucketizeNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    addSupportedPrimDesc({{TensorDescCreatorTypes::ncsp, input_precision},
                          {TensorDescCreatorTypes::ncsp, boundaries_precision}},
                         {{TensorDescCreatorTypes::ncsp, output_precision}},
                         impl_desc_type::ref_any);
}

void MKLDNNBucketizeNode::execute(mkldnn::stream strm) {
    auto precision_mask = getPrecisionMask(input_precision, boundaries_precision, output_precision);

    switch (precision_mask) {
        case getPrecisionMask(Precision::FP32, Precision::FP32, Precision::I32):
            bucketize<PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type>
                    (getParentEdgeAt(0)->getBlob(), getParentEdgeAt(1)->getBlob(), getChildEdgeAt(0)->getBlob());
            break;
        case getPrecisionMask(Precision::FP32, Precision::FP32, Precision::I64):
            bucketize<PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type>
                    (getParentEdgeAt(0)->getBlob(), getParentEdgeAt(1)->getBlob(), getChildEdgeAt(0)->getBlob());
            break;
        case getPrecisionMask(Precision::FP32, Precision::I32, Precision::I32):
            bucketize<PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type>
                    (getParentEdgeAt(0)->getBlob(), getParentEdgeAt(1)->getBlob(), getChildEdgeAt(0)->getBlob());
            break;
        case getPrecisionMask(Precision::FP32, Precision::I32, Precision::I64):
            bucketize<PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type>
                    (getParentEdgeAt(0)->getBlob(), getParentEdgeAt(1)->getBlob(), getChildEdgeAt(0)->getBlob());
            break;
        case getPrecisionMask(Precision::FP32, Precision::I64, Precision::I32):
            bucketize<PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I32>::value_type>
                    (getParentEdgeAt(0)->getBlob(), getParentEdgeAt(1)->getBlob(), getChildEdgeAt(0)->getBlob());
            break;
        case getPrecisionMask(Precision::FP32, Precision::I64, Precision::I64):
            bucketize<PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I64>::value_type>
                    (getParentEdgeAt(0)->getBlob(), getParentEdgeAt(1)->getBlob(), getChildEdgeAt(0)->getBlob());
            break;
        case getPrecisionMask(Precision::I32, Precision::FP32, Precision::I32):
            bucketize<PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type>
                    (getParentEdgeAt(0)->getBlob(), getParentEdgeAt(1)->getBlob(), getChildEdgeAt(0)->getBlob());
            break;
        case getPrecisionMask(Precision::I32, Precision::FP32, Precision::I64):
            bucketize<PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type>
                    (getParentEdgeAt(0)->getBlob(), getParentEdgeAt(1)->getBlob(), getChildEdgeAt(0)->getBlob());
            break;
        case getPrecisionMask(Precision::I32, Precision::I32, Precision::I32):
            bucketize<PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type>
                    (getParentEdgeAt(0)->getBlob(), getParentEdgeAt(1)->getBlob(), getChildEdgeAt(0)->getBlob());
            break;
        case getPrecisionMask(Precision::I32, Precision::I32, Precision::I64):
            bucketize<PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type>
                    (getParentEdgeAt(0)->getBlob(), getParentEdgeAt(1)->getBlob(), getChildEdgeAt(0)->getBlob());
            break;
        case getPrecisionMask(Precision::I32, Precision::I64, Precision::I32):
            bucketize<PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I32>::value_type>
                    (getParentEdgeAt(0)->getBlob(), getParentEdgeAt(1)->getBlob(), getChildEdgeAt(0)->getBlob());
            break;
        case getPrecisionMask(Precision::I32, Precision::I64, Precision::I64):
            bucketize<PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I64>::value_type>
                    (getParentEdgeAt(0)->getBlob(), getParentEdgeAt(1)->getBlob(), getChildEdgeAt(0)->getBlob());
            break;
        case getPrecisionMask(Precision::I64, Precision::FP32, Precision::I32):
            bucketize<PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type>
                    (getParentEdgeAt(0)->getBlob(), getParentEdgeAt(1)->getBlob(), getChildEdgeAt(0)->getBlob());
            break;
        case getPrecisionMask(Precision::I64, Precision::FP32, Precision::I64):
            bucketize<PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type>
                    (getParentEdgeAt(0)->getBlob(), getParentEdgeAt(1)->getBlob(), getChildEdgeAt(0)->getBlob());
            break;
        case getPrecisionMask(Precision::I64, Precision::I32, Precision::I32):
            bucketize<PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type>
                    (getParentEdgeAt(0)->getBlob(), getParentEdgeAt(1)->getBlob(), getChildEdgeAt(0)->getBlob());
            break;
        case getPrecisionMask(Precision::I64, Precision::I32, Precision::I64):
            bucketize<PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type>
                    (getParentEdgeAt(0)->getBlob(), getParentEdgeAt(1)->getBlob(), getChildEdgeAt(0)->getBlob());
            break;
        case getPrecisionMask(Precision::I64, Precision::I64, Precision::I32):
            bucketize<PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I32>::value_type>
                    (getParentEdgeAt(0)->getBlob(), getParentEdgeAt(1)->getBlob(), getChildEdgeAt(0)->getBlob());
            break;
        case getPrecisionMask(Precision::I64, Precision::I64, Precision::I64):
            bucketize<PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I64>::value_type>
                    (getParentEdgeAt(0)->getBlob(), getParentEdgeAt(1)->getBlob(), getChildEdgeAt(0)->getBlob());
            break;
        default:
            IE_THROW() << errorPrefix << " has unsupported precision: " << precision_mask;
    }
}

template <typename T, typename T_BOUNDARIES, typename T_IND>
void MKLDNNBucketizeNode::bucketize(InferenceEngine::Blob::Ptr input, InferenceEngine::Blob::Ptr boundaries, InferenceEngine::Blob::Ptr output) {
    const auto *input_data = input->cbuffer().as<const T *>();
    const auto *boundaries_data = boundaries->cbuffer().as<const T_BOUNDARIES *>();
    auto *output_data = output->buffer().as<T_IND *>();

    if (!with_bins) {
        memset(output_data, 0, num_values * sizeof(T_IND));
        return;
    }

    // boundaries are assumed to be sorted and to have unique elements
    parallel_for(num_values, [&](size_t ind) {
        T value = input_data[ind];
        if (with_right) {
            auto low = std::lower_bound(boundaries_data, boundaries_data + num_bin_values, value);
            output_data[ind] = static_cast<T_IND>(low - boundaries_data);
        } else {
            auto up = std::upper_bound(boundaries_data, boundaries_data + num_bin_values, value);
            output_data[ind] = static_cast<T_IND>(up - boundaries_data);
        }
    });
}

bool MKLDNNBucketizeNode::created() const {
    return getType() == Bucketize;
}

REG_MKLDNN_PRIM_FOR(MKLDNNBucketizeNode, Bucketize)
