//*****************************************************************************
// Copyright 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;

using Attrs = op::v6::ExperimentalDetectronDetectionOutput::Attributes;
using ExperimentalDetection = op::v6::ExperimentalDetectronDetectionOutput;

TEST(type_prop, detectron_detection_output)
{
    Attrs attrs;
    attrs.class_agnostic_box_regression = false;
    attrs.deltas_weights = {10.0f, 10.0f, 5.0f, 5.0f};
    attrs.max_delta_log_wh = 4.135166645050049f;
    attrs.max_detections_per_image = 100;
    attrs.nms_threshold = 0.5f;
    attrs.num_classes = 81;
    attrs.post_nms_count = 2000;
    attrs.score_threshold = 0.05000000074505806f;

    size_t rois_num = static_cast<size_t>(attrs.max_detections_per_image);

    auto rois = std::make_shared<op::Parameter>(element::f32, Shape{1000, 4});
    auto deltas = std::make_shared<op::Parameter>(element::f32, Shape{1000, 324});
    auto scores = std::make_shared<op::Parameter>(element::f32, Shape{1000, 81});
    auto im_info = std::make_shared<op::Parameter>(element::f32, Shape{1, 3});

    auto detection = std::make_shared<ExperimentalDetection>(rois, deltas, scores, im_info, attrs);

    ASSERT_EQ(detection->get_output_element_type(0), element::f32);
    ASSERT_EQ(detection->get_output_element_type(1), element::i32);
    ASSERT_EQ(detection->get_output_element_type(2), element::f32);
    ASSERT_EQ(detection->get_output_element_type(3), element::i32);

    EXPECT_EQ(detection->get_output_shape(0), Shape{rois_num, 4}));
    EXPECT_EQ(detection->get_output_shape(1), Shape{rois_num}));
    EXPECT_EQ(detection->get_output_shape(2), (Shape{rois_num}));
    EXPECT_EQ(detection->get_output_shape(3), (Shape{rois_num}));
}