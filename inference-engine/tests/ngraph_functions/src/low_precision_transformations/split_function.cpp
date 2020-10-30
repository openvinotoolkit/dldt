// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/split_function.hpp"

#include "ngraph_functions/subgraph_builders.hpp"
#include "low_precision/network_helper.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"


namespace ngraph {
namespace builder {
namespace subgraph {
    std::shared_ptr<ngraph::Function> SplitFunction::getOriginal(
        const ngraph::Shape& inputShape,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const int64_t splitedAxis,
        const size_t numSplits) {
        const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
            precisionBeforeDequantization,
            ngraph::Shape(inputShape));
        input->set_friendly_name("input");

        const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
        const auto constant = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
        const std::shared_ptr<Node> split = std::make_shared<ngraph::opset1::Split>(dequantizationOp, constant, numSplits);
        split->set_friendly_name("split");

        ngraph::ResultVector results;
        for (size_t i = 0; i < numSplits; ++i) {
            results.push_back(std::make_shared<ngraph::opset1::Result>(split->output(i)));
            results[i]->set_friendly_name("result" + std::to_string(i + 1));
        }
        return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SplitFunction");
    }

std::shared_ptr<ngraph::Function> SplitFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize,
    int64_t splitedAxis, size_t numSplit) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(originalFunctionPrecision, inputShape);

    const auto fq = fakeQuantize.empty() ? nullptr :
        ngraph::builder::makeFakeQuantize(
            input,
            originalFunctionPrecision,
            fakeQuantize.quantizationLevel,
            fakeQuantize.constantShape,
            fakeQuantize.inputLowValues,
            fakeQuantize.inputHighValues,
            fakeQuantize.outputLowValues,
            fakeQuantize.outputHighValues);

    auto constant = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
    const std::shared_ptr<ngraph::opset1::Split> split = std::make_shared<ngraph::opset1::Split>(fq, constant, numSplit);

    ngraph::ResultVector results;
    for (size_t i = 0; i < numSplit; ++i) {
        results.push_back(std::make_shared<ngraph::opset1::Result>(split->output(i)));
    }
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SplitFunction");
}

std::shared_ptr<ngraph::Function> SplitFunction::getReference(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precision,
    const std::vector<ngraph::builder::subgraph::DequantizationOperations>& dequantizationAfter,
    const int64_t splitedAxis,
    const size_t numSplit) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precision,
        ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    std::shared_ptr<ngraph::opset1::Split> split;
    const auto constant = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
    split = std::make_shared<ngraph::opset1::Split>(input, constant, numSplit);
    split->set_friendly_name("split");

    ngraph::ResultVector results;
    ngraph::Output<ngraph::Node> lastNode;
    for (size_t i = 0; i < numSplit; ++i) {
        if (dequantizationAfter.empty()) {
            lastNode = split->output(i);
        } else {
            lastNode = makeDequantization(split->output(i), dequantizationAfter[i], i);
            lastNode.get_node_shared_ptr()->set_friendly_name("split." + std::to_string(i));
        }
        results.push_back(std::make_shared<ngraph::opset1::Result>(lastNode));
        results[i]->set_friendly_name("result" + std::to_string(i + 1));
    }

    if (!dequantizationAfter.empty()) {
        split->set_friendly_name("split_original");
    }

    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SplitTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
