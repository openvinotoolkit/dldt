// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "functional_test_utils/layer_test_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/common_utils.hpp"

namespace BehaviorTestsDefinitions {

enum class setType {
    INPUT,
    OUTPUT,
    BOTH
};

std::ostream& operator<<(std::ostream & os, setType type);

using SetBlobParams = std::tuple<InferenceEngine::Precision,   // precision in CNNNetwork
                                 setType,                      // type for which blob is set
                                 std::string>;                 // Device name

class SetBlobTest : public testing::WithParamInterface<SetBlobParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SetBlobParams> obj);
    void Infer() override;

protected:
    void SetUp() override;

private:
    InferenceEngine::Precision precision;
    setType type;
};

} // namespace BehaviorTestsDefinitions
