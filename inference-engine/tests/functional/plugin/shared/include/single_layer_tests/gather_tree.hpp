// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

using GatherTreeParamsTuple = typename std::tuple<
        std::vector<size_t>,               // Input tensors shape
        InferenceEngine::Precision,        // Network precision
        std::string>;                      // Device name

class GatherTreeLayerTest : public testing::WithParamInterface<GatherTreeParamsTuple>,
                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherTreeParamsTuple> &obj);

protected:
    void SetUp() override;

private:
    enum InputAxis {
        MAX_TIME,
        BATCH_SIZE,
        BEAM_WIDTH
    };
};

} // namespace LayerTestsDefinitions