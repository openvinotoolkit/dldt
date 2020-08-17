// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API HSwishFusion;
class TRANSFORMATIONS_API HSwishFusionWithRelu;
class TRANSFORMATIONS_API HSwishFusionWithoutRelu;


}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces various sub-graphs with a HSwish op.
 */
class ngraph::pass::HSwishFusion: public ngraph::pass::GraphRewrite {
public:
    HSwishFusion() {
        add_matcher<ngraph::pass::HSwishFusionWithRelu>();
        add_matcher<ngraph::pass::HSwishFusionWithoutRelu>();
    }
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph x * (min(Relu(x + 3), 6) / 6) with a HSwish op.
 */
 class ngraph::pass::HSwishFusionWithRelu: public ngraph::pass::MatcherPass {
public:
    HSwishFusionWithRelu();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph x * (min(max(x + 3, 0), 6) / 6) with a HSwish op.
 */
 class ngraph::pass::HSwishFusionWithoutRelu: public ngraph::pass::MatcherPass {
public:
    HSwishFusionWithoutRelu();
};
