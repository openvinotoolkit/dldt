// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/type_relaxed.hpp>

#include "low_precision/network_helper.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"

#include "lpt_ngraph_functions/avg_pool_function.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> AvgPoolFunction::getOriginal(
    const ngraph::element::Type inputPrecision,
    const ngraph::Shape& inputShape,
    const bool addFQ,
    const std::string additionalLayer,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent = input;

    const auto dequantization = makeDequantization(input, dequantizationBefore);

    const std::shared_ptr<ngraph::Node> avgPool = std::make_shared<ngraph::opset1::AvgPool>(
        dequantization,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        true,
        op::RoundingType::FLOOR);

    std::shared_ptr<Node> lastLayer = avgPool;
    if (additionalLayer == "maxpool") {
        lastLayer = std::make_shared<ngraph::opset1::MaxPool>(
            lastLayer,
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 0, 0 },
            Shape{ 2, 2 },
            op::RoundingType::FLOOR);
    }

    if (addFQ) {
        lastLayer = ngraph::builder::makeFakeQuantize(
            lastLayer, ngraph::element::f32, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
    }

    lastLayer->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastLayer) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "AvgPoolTransformation");
}

std::shared_ptr<ngraph::Function> AvgPoolFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(originalFunctionPrecision, ngraph::Shape(inputShape));

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        input, originalFunctionPrecision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);

    const std::shared_ptr<ngraph::Node> avgPool = std::make_shared<ngraph::opset1::AvgPool>(
        fakeQuantize,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        true,
        op::RoundingType::FLOOR);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(avgPool) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "AvgPoolTransformation");
}

std::shared_ptr<ngraph::Function> AvgPoolFunction::getReference(
    const ngraph::element::Type inputPrecision,
    const ngraph::Shape& inputShape,
    const bool addFQ,
    const std::string additionalLayer,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, ngraph::Shape(inputShape));
    const auto deqBefore = makeDequantization(input, dequantizationBefore);

    const std::shared_ptr<ngraph::Node> avgPool = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::AvgPool>>(
        deqBefore,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        true,
        op::RoundingType::FLOOR);
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(avgPool, precisionAfterOperation);

    std::shared_ptr<Node> lastLayer = avgPool;
    if (additionalLayer == "maxpool") {
        lastLayer = std::make_shared<ngraph::opset1::MaxPool>(
            lastLayer,
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 0, 0 },
            Shape{ 2, 2 },
            op::RoundingType::FLOOR);
    }

    lastLayer = makeDequantization(lastLayer, dequantizationAfter);

    if (addFQ) {
        lastLayer = ngraph::builder::makeFakeQuantize(
            lastLayer, ngraph::element::f32, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
    }

    lastLayer->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastLayer) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "AvgPoolTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
