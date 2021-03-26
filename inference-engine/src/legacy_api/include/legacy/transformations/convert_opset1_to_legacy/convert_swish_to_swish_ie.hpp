// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {
    class INFERENCE_ENGINE_API_CLASS(ConvertSwishToSwishIEMatcher);
}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertSwishToSwishIEMatcher: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertSwishToSwishIEMatcher();
};
