// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "transformation_context.hpp"
#include "layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API WeightableLayerTransformation : public LayerTransformation{
public:
    WeightableLayerTransformation(const Params& params);
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    bool isQuantized(std::shared_ptr<Node> layer, bool isReshape) const noexcept;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;

protected:
    DataPrecision decomposeFakeQuantizeForWeightsPath(std::shared_ptr<Node> weightableLayer, const bool supportAsymmetricQuantization) const;
    static bool isGroup(const std::shared_ptr<Node>& node);
    static bool isDepthwise(const std::shared_ptr<Node>& node);
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
