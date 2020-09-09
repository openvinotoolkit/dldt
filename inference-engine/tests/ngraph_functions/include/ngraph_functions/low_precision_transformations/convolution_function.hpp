// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_weights.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class ConvolutionFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type inputPrecision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        std::shared_ptr<ngraph::opset1::Constant> weights,
        const ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights);

    static std::shared_ptr<ngraph::Function> getOriginalWithIncorrectWeights(
        const ngraph::Shape& inputShape,
        ngraph::element::Type precision,
        ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData,
        bool isCrorrect);

    static std::shared_ptr<ngraph::Function> getReferenceWithIncorrectWeights(
        const ngraph::Shape& inputShape,
        ngraph::element::Type precision,
        ngraph::element::Type dataPrecision,
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData,
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore,
        ngraph::element::Type weightsPrecision,
        std::vector<float> weightsValues,
        ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter,
        bool isCorrect);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type inputPrecision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        std::shared_ptr<ngraph::opset1::Constant> weights,
        const ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
        const ngraph::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
        const ngraph::element::Type precisionAfterDequantization);
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
