// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

typedef std::tuple<
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::map<std::string, std::string>  //Configuration
> concatQuantizationParams;

namespace LayerTestsDefinitions {

class ConcatQuantization : public testing::WithParamInterface<concatQuantizationParams>,
                        public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<concatQuantizationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
