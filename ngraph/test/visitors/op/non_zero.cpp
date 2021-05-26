// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset3.hpp"

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, non_zero_op_default)
{
    NodeBuilder::get_ops().register_factory<opset3::NonZero>();
    const auto data_node = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto non_zero = make_shared<op::NonZero>(data_node);

    NodeBuilder builder(non_zero);
    const auto expected_attr_count = 1;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(non_zero->get_output_type(), element::i64);
}

TEST(attributes, non_zero_op_i32)
{
    NodeBuilder::get_ops().register_factory<opset3::NonZero>();
    const auto data_node = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto non_zero = make_shared<op::NonZero>(data_node, "i32");

    NodeBuilder builder(non_zero);
    const auto expected_attr_count = 1;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(non_zero->get_output_type(), element::i32);
}
