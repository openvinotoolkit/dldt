﻿// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/mat_mul.hpp"

#include <numeric>
#include <memory>
#include <string>
#include <vector>

#include "transformations/low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

bool MatMulTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<ngraph::opset1::MatMul> matMul = as_type_ptr<ngraph::opset1::MatMul>(m.get_match_root());
    if ((matMul == nullptr) || !canBeTransformed(context, matMul)) {
        return false;
    }

    matMul = as_type_ptr<ngraph::opset1::MatMul>(separateInStandaloneBranch(matMul));

    FakeQuantizeDequantization dequantization2 = ngraph::pass::low_precision::NetworkHelper::getDequantization(matMul, 1);
    if (dequantization2.empty()) {
        const std::shared_ptr<opset1::FakeQuantize> fakeQuantize =
            as_type_ptr<opset1::FakeQuantize>(dequantization2.data.get_node_shared_ptr());
        if (fakeQuantize != nullptr) {
            const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fakeQuantize);
            const DataPrecision dataPrecision = getDataPrecision(fakeQuantize, quantizationDetails, true);

            auto tuple = NetworkHelper::decomposeFakeQuantize(
                fakeQuantize,
                dataPrecision.precision,
                dataPrecision.min,
                dataPrecision.max,
                dataPrecision.hasZeroPoint,
                updatePrecisions);

            dequantization2 = ngraph::pass::low_precision::NetworkHelper::getDequantization(matMul, 1);
        }
    }

    const FakeQuantizeDequantization dequantization1 = ngraph::pass::low_precision::NetworkHelper::getDequantization(matMul, 0);
    const std::shared_ptr<opset1::MatMul> newMatMul = std::make_shared<ngraph::op::TypeRelaxed<opset1::MatMul>>(
        std::vector<element::Type>({ element::f32, element::f32 }), std::vector<element::Type>({}),
        ngraph::op::TemporaryReplaceOutputType(dequantization1.data, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(dequantization2.data, element::f32).get(),
        matMul->get_transpose_a(),
        matMul->get_transpose_b());
    NetworkHelper::setOutDataPrecisionForTypeRelaxed(newMatMul, matMul->get_output_element_type(0));


    auto transpose = [](const std::shared_ptr<Node>& node) -> std::shared_ptr<Node> {
        const Shape outputShape = node->get_output_shape(0);

        std::vector<size_t> transposeConstant(outputShape.size());
        std::iota(transposeConstant.begin(), transposeConstant.end(), 0);
        std::swap(*(transposeConstant.end() - 1), *(transposeConstant.end() - 2));

        std::shared_ptr<Node> transposedConstant = fold<ngraph::opset1::Transpose>(
            node,
            opset1::Constant::create(element::i64, Shape{ transposeConstant.size() }, transposeConstant));
        return transposedConstant;
    };

    const std::shared_ptr<Node> const1 = matMul->get_transpose_a() ?
        transpose(dequantization1.multiply->get_input_node_shared_ptr(1)) :
        dequantization1.multiply->get_input_node_shared_ptr(1);

    const std::shared_ptr<Node> const2 = matMul->get_transpose_b() ?
        transpose(dequantization2.multiply->get_input_node_shared_ptr(1)) :
        dequantization2.multiply->get_input_node_shared_ptr(1);

    const std::shared_ptr<opset1::Multiply> newMultiply = std::make_shared<DequantizationMultiply>(
        newMatMul,
        NetworkHelper::toScalarIfPossible(
            fold<ngraph::opset1::Multiply>(
                NetworkHelper::toScalar(as_type_ptr<opset1::Constant>(const1)),
                const2)));
    replace_node(matMul, newMultiply);

    updateOutput(context, newMultiply, matMul);
    return true;
}

void MatMulTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::MatMul>({ make_op_label<ngraph::opset1::Multiply>(), make_op_label<ngraph::opset1::Multiply>() }));

    addPattern(
        pass,
        context,
        make_op_pattern<opset1::MatMul>({ make_op_label<ngraph::opset1::Multiply>(), make_op_label<ngraph::opset1::FakeQuantize>() }));
}

bool MatMulTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

bool MatMulTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    if (!LayerTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    if (!canSubtractBeHandled(layer)) {
        return false;
    }

    const auto dequantization1 = ngraph::pass::low_precision::NetworkHelper::getDequantization(layer);

    if (!NetworkHelper::isScalarLike(
            as_type_ptr<opset1::Constant>(dequantization1.multiply->get_input_node_shared_ptr(1)))) {
        return false;
    }

    return true;
}
