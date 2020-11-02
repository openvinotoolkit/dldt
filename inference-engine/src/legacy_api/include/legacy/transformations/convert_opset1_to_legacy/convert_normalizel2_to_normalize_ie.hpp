// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>


namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertNormalizeL2WithMulToNormalizeIE);
class INFERENCE_ENGINE_API_CLASS(ConvertNormalizeL2ToLegacyMatcher);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertNormalizeL2WithMulToNormalizeIE: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertNormalizeL2WithMulToNormalizeIE();
};

class ngraph::pass::ConvertNormalizeL2ToLegacyMatcher: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertNormalizeL2ToLegacyMatcher();
};
