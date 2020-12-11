// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/relu_shape_of.hpp"

namespace LayerTestsDefinitions {

TEST_P(ReluShapeOfSubgraphTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions