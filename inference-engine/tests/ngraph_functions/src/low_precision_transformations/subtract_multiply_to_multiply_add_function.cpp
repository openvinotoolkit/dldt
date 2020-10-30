// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/subtract_multiply_to_multiply_add_function.hpp"

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "low_precision/common/dequantization_op.hpp"

using namespace ngraph::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> SubtractMultiplyToMultiplyAddFunction::getOriginal(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization,
    const ngraph::element::Type precisionAfterDequantization) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    dequantizationOp->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantizationOp) };
    results[0]->set_friendly_name("result");
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SubtractMultiplyToMultiplyAddFunction");
}

std::shared_ptr<ngraph::Function> SubtractMultiplyToMultiplyAddFunction::getOriginal(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precision,
    const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precision,
        ngraph::Shape(inputShape));

    const std::shared_ptr<Node> fq = makeFakeQuantize(input, precision, fqOnData);

    const std::shared_ptr<ngraph::opset1::Reshape> reshape1 = std::make_shared<ngraph::opset1::Reshape>(
        fq,
        std::make_shared<ngraph::opset1::Constant>(
            ngraph::element::i64,
            Shape({ 3 }),
            std::vector<int64_t>({ static_cast<int64_t>(inputShape[0]), static_cast<int64_t>(inputShape[1]), -1 })),
        false);

    const std::shared_ptr<ngraph::opset1::Reshape> reshape2 = std::make_shared<ngraph::opset1::Reshape>(
        reshape1,
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, Shape({ 4 }), inputShape),
        false);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape2) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SubtractMultiplyToMultiplyAddFunction");
}

std::shared_ptr<ngraph::Function> SubtractMultiplyToMultiplyAddFunction::getReference(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization,
    const ngraph::element::Type precisionAfterDequantization,
    const ngraph::builder::subgraph::Multiply& multiply,
    const ngraph::builder::subgraph::Add& add) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    std::shared_ptr<Node> parent = dequantizationOp;

    std::shared_ptr<ngraph::Node> mul;
    if (!multiply.empty()) {
        mul = makeElementwise<DequantizationMultiply>(parent, multiply);
        ngraph::pass::low_precision::NetworkHelper::setDequantizationName(parent, mul);
        parent = mul;
    }

    if (!add.empty()) {
        const auto addNode = makeElementwise<DequantizationAdd>(parent, add);
        ngraph::pass::low_precision::NetworkHelper::setDequantizationName(parent, addNode);
        mul->set_friendly_name("output_original");
        parent = addNode;
    }
    parent->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(parent) };
    results[0]->set_friendly_name("result");
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SubtractMultiplyToMultiplyAddFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
