// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <shared_test_classes/subgraph/first_connect_input_concat.hpp>

namespace LayerTestsDefinitions {

TEST_P(ConcatFirstInputTest, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
