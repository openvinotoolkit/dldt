// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, clamp_basic_f32)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 32, 32});
    auto clamp = make_shared<op::Clamp>(data, 0.0, 2.1);

    ASSERT_EQ(clamp->get_element_type(), element::f32);
    ASSERT_EQ(clamp->get_min(), 0.0);
    ASSERT_EQ(clamp->get_max(), 2.1);
    ASSERT_EQ(clamp->get_output_shape(0), (Shape{1, 32, 32}));
}

TEST(type_prop, clamp_basic_i32)
{
    auto data = make_shared<op::Parameter>(element::i32, Shape{1, 32, 32});
    auto clamp = make_shared<op::Clamp>(data, 0.0, 2.1);

    ASSERT_EQ(clamp->get_element_type(), element::i32);
    ASSERT_EQ(clamp->get_min(), 0.0);
    ASSERT_EQ(clamp->get_max(), 2.1);
    ASSERT_EQ(clamp->get_output_shape(0), (Shape{1, 32, 32}));
}

TEST(type_prop, clamp_shape_static_rank)
{
    auto data = make_shared<op::Parameter>(
        element::f16, PartialShape{Dimension::dynamic(), Dimension::dynamic(), 32});
    auto clamp = make_shared<op::Clamp>(data, -2.1, 2.1);

    ASSERT_EQ(clamp->get_element_type(), element::f16);
    ASSERT_EQ(clamp->get_min(), -2.1);
    ASSERT_EQ(clamp->get_max(), 2.1);
    ASSERT_EQ(clamp->get_output_partial_shape(0),
              (PartialShape{Dimension::dynamic(), Dimension::dynamic(), 32}));
}

TEST(type_prop, clamp_shape_dynamic)
{
    auto data = make_shared<op::Parameter>(element::u16, PartialShape::dynamic());
    auto clamp = make_shared<op::Clamp>(data, 1.5, 15.0);

    ASSERT_EQ(clamp->get_element_type(), element::u16);
    ASSERT_EQ(clamp->get_min(), 1.5);
    ASSERT_EQ(clamp->get_max(), 15.0);
    ASSERT_EQ(clamp->get_output_partial_shape(0), (PartialShape::dynamic()));
}

TEST(type_prop, clamp_invalid_element_type)
{
    auto data = make_shared<op::Parameter>(element::boolean, Shape{2, 2});

    try
    {
        auto clamp = make_shared<op::Clamp>(data, 0.5, 5.5);
        // Input element type is boolean
        FAIL() << "Invalid boolean element type for input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Input element type must be numeric");
    }
    catch (...)
    {
        FAIL() << "Numeric element type node validation check failed for unexpected reason";
    }
}

TEST(type_prop, clamp_invalid_attributes)
{
    auto data = make_shared<op::Parameter>(element::f64, Shape{2, 2});

    try
    {
        auto clamp = make_shared<op::Clamp>(data, 1.0, 1.0);
        // Attribute 'max' not greater than 'min'
        FAIL() << "Attribute 'min' equal to 'max' not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Attribute 'min' must be less than 'max'");
    }
    catch (...)
    {
        FAIL() << "'min' and 'max' attributes node validation check failed for unexpected reason";
    }
}
