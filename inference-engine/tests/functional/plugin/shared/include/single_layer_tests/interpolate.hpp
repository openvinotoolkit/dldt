// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/interpolate.hpp"

namespace LayerTestsDefinitions {

TEST_P(InterpolateLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(InterpolateLayerTestWithConfig, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
