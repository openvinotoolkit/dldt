// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer_test_classes/broadcast.hpp"

namespace LayerTestsDefinitions {

TEST_P(BroadcastLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
