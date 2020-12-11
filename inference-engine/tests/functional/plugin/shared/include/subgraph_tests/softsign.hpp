// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/softsign.hpp"

namespace LayerTestsDefinitions {

TEST_P(SoftsignTest, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
