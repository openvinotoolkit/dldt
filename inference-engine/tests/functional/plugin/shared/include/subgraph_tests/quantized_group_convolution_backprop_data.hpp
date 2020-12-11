// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/quantized_group_convolution_backprop_data.hpp"

namespace LayerTestsDefinitions {

TEST_P(QuantGroupConvBackpropDataLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions