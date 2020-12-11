// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"


typedef std::tuple<
        std::vector<size_t>,               // Logits shapes
        std::vector<int>,                  // logits lenght
        std::vector<std::vector<int>>,     // labels
        std::vector<int>,                  // labels length
        int,                               // blank index
        bool,                              // preprocessCollapseRepeated
        bool,                              // ctcMergeRepeated
        bool                               // Unique
> CTCLossParamsSubset;

typedef std::tuple<
        CTCLossParamsSubset,
        InferenceEngine::Precision,        // Float point precision
        InferenceEngine::Precision,        // Integer precision
        LayerTestsUtils::TargetDevice      // Device name
> CTCLossParams;

namespace LayerTestsDefinitions {

class CTCLossLayerTest : public testing::WithParamInterface<CTCLossParams>,
                        public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CTCLossParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
