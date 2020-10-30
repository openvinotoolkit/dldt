// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/reshape_fully_connected_function.hpp"

#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>


#include <ngraph/opsets/opset1.hpp>
#include <legacy/ngraph_ops/fully_connected.hpp>
#include <legacy/ngraph_ops/scaleshift.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/common/dequantization_op.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> ReshapeFullyConnectedFunction::getOriginal(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type inputPrecision1,
    const ngraph::element::Type inputPrecision2,
    const ngraph::element::Type inputPrecision3,
    const ngraph::Shape& outputShape,
    const ngraph::element::Type outputPrecision) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision1, inputShape);
    input->set_friendly_name("input");

    const auto weightsShape = Shape{ outputShape[1], inputShape[1] };
    const auto weights = std::make_shared<opset1::Constant>(inputPrecision2, weightsShape, std::vector<float>(shape_size(weightsShape), 1.f));
    const auto bias = std::make_shared<opset1::Constant>(inputPrecision3, Shape{ inputShape[1] }, 0.f);

    const std::shared_ptr<op::FullyConnected> fullyConnected = std::make_shared<op::FullyConnected>(input, weights, bias, outputShape, outputPrecision);
    fullyConnected->set_friendly_name("fullyConnected");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fullyConnected) };
    results[0]->set_friendly_name("result");
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ReshapeFullyConnectedFunction");
}

std::shared_ptr<ngraph::Function> ReshapeFullyConnectedFunction::getReference(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type inputPrecision1,
    const ngraph::element::Type inputPrecision2,
    const ngraph::element::Type inputPrecision3,
    const ngraph::Shape& outputShape,
    const ngraph::element::Type outputPrecision) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision1, inputShape);
    input->set_friendly_name("input");

    std::vector<int64_t> reshapeShape{ -1, static_cast<int64_t>(inputShape.back()) };
    auto reshape = std::make_shared<opset1::Reshape>(input, opset1::Constant::create(element::i64, Shape{ 2 }, reshapeShape), true);
    reshape->set_friendly_name("fullyConnected/Reshape");

    const auto weightsShape = Shape{ outputShape[1], inputShape[1] };
    const auto weights = std::make_shared<opset1::Constant>(inputPrecision2, weightsShape, std::vector<float>(shape_size(weightsShape), 1.f));
    const auto bias = std::make_shared<opset1::Constant>(inputPrecision3, Shape{ inputShape[1] }, 0.f);

    const std::shared_ptr<op::FullyConnected> fullyConnected = std::make_shared<op::FullyConnected>(reshape, weights, bias, outputShape, outputPrecision);
    fullyConnected->set_friendly_name("fullyConnected");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fullyConnected) };
    results[0]->set_friendly_name("result");
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ReshapeFullyConnectedFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
