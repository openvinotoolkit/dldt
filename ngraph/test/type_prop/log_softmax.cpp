// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, log_softmax)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6});
    auto log_softmax_func = make_shared<op::v5::LogSoftmax>(data, 1);
    EXPECT_EQ(log_softmax_func->get_element_type(), element::f32);
    EXPECT_EQ(log_softmax_func->get_shape(), (Shape{1, 3, 6}));
}

TEST(type_prop, log_softmax_incorrect_axis)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6});

    try
    {
        auto log_softmax_func = make_shared<op::v5::LogSoftmax>(data, 3);
        FAIL() << "LogSoftmax node was created with incorrect axis.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Reduction axis (3) is out of bounds");
    }
}

TEST(type_prop, log_softmax_partial)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto log_softmax_func = make_shared<op::v5::LogSoftmax>(data, 1);
    EXPECT_EQ(log_softmax_func->get_element_type(), element::f32);
    ASSERT_TRUE(log_softmax_func->get_output_partial_shape(0).same_scheme(
        (PartialShape{1, Dimension::dynamic(), 6})));

    // rank unknown
    auto log_softmax_partial = make_shared<op::v5::LogSoftmax>(
        make_shared<op::Parameter>(element::f32, PartialShape::dynamic()));
    ASSERT_TRUE(
        log_softmax_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, log_softmax_partial_static_rank)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto log_softmax_func = make_shared<op::v5::LogSoftmax>(data, 1);
    EXPECT_EQ(log_softmax_func->get_element_type(), element::f32);
    ASSERT_TRUE(log_softmax_func->get_output_partial_shape(0).same_scheme(
        (PartialShape{1, Dimension::dynamic(), 6})));
    ASSERT_TRUE(log_softmax_func->get_output_partial_shape(0).rank().is_static());
}
