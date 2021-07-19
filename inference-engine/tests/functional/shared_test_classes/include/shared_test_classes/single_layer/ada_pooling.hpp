// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using adapoolParams = std::tuple<
        std::vector<size_t>,                // feature map shape
        std::vector<int>,                   // pooled spatial shape
        std::string,                        // pooling mode
        InferenceEngine::Precision,         // net precision
        LayerTestsUtils::TargetDevice>;     // device name

class AdaPoolLayerTest : public testing::WithParamInterface<adapoolParams>,
                          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<adapoolParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
