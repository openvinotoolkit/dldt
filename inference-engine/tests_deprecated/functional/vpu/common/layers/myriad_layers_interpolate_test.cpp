// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_interpolate_test.hpp"

INSTANTIATE_TEST_CASE_P(
	accuracy, myriadInterpolateLayerTests_nightly,
	::testing::Combine(
		::testing::ValuesIn(s_InterpolateInput),
		::testing::Values<Factor>(2.0f, 0.5f),
		::testing::Values<Antialias>(false),
		// ::testing::Values<cube_coeff>(-0.75),
		// ::testing::Values<batch>(1),
		// ::testing::Values<type>(0),
		::testing::Values<nearestMode>(0),
		::testing::Values<shapeCalcMode>(1),
		::testing::Values<coordTransMode>(0),
		// ::testing::Values<pads_begin>(0),
		// ::testing::Values<pads_end>(0),
		// ::testing::Values<sizes>(-1),
		::testing::Values<InterpolateAxis>(0, 1),
		::testing::Values<InterpolateScales>(2, 2),
		::testing::Values<HwOptimization>(false, true),
		::testing::Values(""))
);

// #ifdef VPU_HAS_CUSTOM_KERNELS

// INSTANTIATE_TEST_CASE_P(
// 	accuracy_custom, myriadInterpolateLayerTests_nightly,
// 	::testing::Combine(
// 		::testing::ValuesIn(s_InterpolateInput),
// 		::testing::Values<Factor>(2.0f),
// 		::testing::Values<Antialias>(false, true),
// 		::testing::Values<HwOptimization>(false, true),
// 		::testing::Values(s_CustomConfig[1]))
// );

// #endif
