// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, selu_basic_inference_f32_3D)
{
    const auto param = make_shared<op::Parameter>(element::f32, Shape{1, 32, 32});
    const auto alpha = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto lambda = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto selu = make_shared<op::Selu>(param, alpha, lambda);

    ASSERT_EQ(selu->get_element_type(), element::f32);
    ASSERT_EQ(selu->get_shape(), (Shape{1, 32, 32}));
}

TEST(type_prop, selu_basic_inference_f16_3D)
{
    const auto param = make_shared<op::Parameter>(element::f16, Shape{1, 32, 32});
    const auto alpha = make_shared<op::Parameter>(element::f16, Shape{1});
    const auto lambda = make_shared<op::Parameter>(element::f16, Shape{1});
    const auto selu = make_shared<op::Selu>(param, alpha, lambda);

    ASSERT_EQ(selu->get_element_type(), element::f16);
    ASSERT_EQ(selu->get_shape(), (Shape{1, 32, 32}));
}

TEST(type_prop, selu_basic_inference_f32_5D)
{
    const auto param = make_shared<op::Parameter>(element::f32, Shape{12, 135, 221, 31, 15});
    const auto alpha = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto lambda = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto selu = make_shared<op::Selu>(param, alpha, lambda);

    ASSERT_EQ(selu->get_element_type(), element::f32);
    ASSERT_EQ(selu->get_shape(), (Shape{12, 135, 221, 31, 15}));
}

TEST(type_prop, selu_basic_inference_f16_5D)
{
    const auto param = make_shared<op::Parameter>(element::f16, Shape{12, 135, 221, 31, 15});
    const auto alpha = make_shared<op::Parameter>(element::f16, Shape{1});
    const auto lambda = make_shared<op::Parameter>(element::f16, Shape{1});
    const auto selu = make_shared<op::Selu>(param, alpha, lambda);

    ASSERT_EQ(selu->get_element_type(), element::f16);
    ASSERT_EQ(selu->get_shape(), (Shape{12, 135, 221, 31, 15}));
}

TEST(type_prop, selu_incompatible_input_type_boolean)
{
    const auto param = make_shared<op::Parameter>(element::boolean, Shape{1, 32, 32});
    const auto alpha = make_shared<op::Parameter>(element::f16, Shape{1});
    const auto lambda = make_shared<op::Parameter>(element::f16, Shape{1});
    ASSERT_THROW(std::make_shared<op::Selu>(param, alpha, lambda), ngraph::NodeValidationFailure);
}

TEST(type_prop, selu_incompatible_input_type_i32)
{
    const auto param = make_shared<op::Parameter>(element::i32, Shape{1, 32, 32});
    const auto alpha = make_shared<op::Parameter>(element::i32, Shape{1});
    const auto lambda = make_shared<op::Parameter>(element::i32, Shape{1});
    ASSERT_THROW(std::make_shared<op::Selu>(param, alpha, lambda), ngraph::NodeValidationFailure);
}

TEST(type_prop, selu_incompatible_input_type_u16)
{
    const auto param = make_shared<op::Parameter>(element::u16, Shape{1, 32, 32});
    const auto alpha = make_shared<op::Parameter>(element::u16, Shape{1});
    const auto lambda = make_shared<op::Parameter>(element::u16, Shape{1});
    ASSERT_THROW(std::make_shared<op::Selu>(param, alpha, lambda), ngraph::NodeValidationFailure);
}

TEST(type_prop, selu_dynamic_rank_input_shape_2D)
{
    const PartialShape param_shape{Dimension::dynamic(), 10};
    const auto param = std::make_shared<op::Parameter>(element::f32, param_shape);
    const auto alpha = make_shared<op::Parameter>(element::f32, Shape{2, 1});
    const auto lambda = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto op = std::make_shared<op::Selu>(param, alpha, lambda);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{Dimension(), 10}));
}

TEST(type_prop, selu_dynamic_rank_input_shape_3D)
{
    const PartialShape param_shape{100, Dimension::dynamic(), 58};
    const auto param = std::make_shared<op::Parameter>(element::f32, param_shape);
    const auto alpha = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto lambda = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto op = std::make_shared<op::Selu>(param, alpha, lambda);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{100, Dimension(), 58}));
}

TEST(type_prop, selu_dynamic_rank_input_shape_full)
{
    const auto param = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto alpha = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto lambda = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto op = std::make_shared<op::Selu>(param, alpha, lambda);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}
