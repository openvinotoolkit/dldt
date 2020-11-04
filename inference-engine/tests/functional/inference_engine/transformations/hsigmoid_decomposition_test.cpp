// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/op_conversions/hsigmoid_decomposition.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, HSigmoidDecompositionTest) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(1));
        auto hsigmoid = std::make_shared<ngraph::opset5::HSigmoid>(input);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{hsigmoid}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::HSigmoidDecomposition>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset5::Add>(input, add_constant);
        auto relu = std::make_shared<ngraph::opset5::Relu>(add);
        auto min_constant = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{}, {6.0});
        auto min = std::make_shared<ngraph::opset5::Minimum>(relu, min_constant);
        auto mul_constant = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{}, {(1.0/6.0)});  // const(1/6)
        auto mul = std::make_shared<ngraph::opset5::Multiply>(min, mul_constant);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
