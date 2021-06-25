// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/reorder_activation_and_pooling.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>


namespace testing {

// Variant Convolution -> Add -> Activation -> MaxPool : ConvAddAct

template <typename ActivationT, typename ... Args>
std::shared_ptr<ngraph::Function> createFunctionConvAddAct(Args&& ... args) {
    auto input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 3, 64, 64});
    auto input_params_add = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 3, 64, 64});

    auto weights = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{3, 3, 1, 1}, {1});
    auto bias = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{3, 1, 1}, {1});
    auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                                weights,
                                                                                ngraph::Strides{1, 1},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::Strides{1, 1});

    auto add_operation = std::make_shared<ngraph::opset7::Add>(convolution_operation,
                                                               input_params_add);

    auto activation = std::make_shared<ActivationT>(add_operation, std::forward<Args>(args) ...);

    auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(activation,
                                                                        ngraph::Strides{1, 1},
                                                                        ngraph::Shape{1, 1},
                                                                        ngraph::Shape{1, 1},
                                                                        ngraph::Shape{1, 1});

    auto result = std::make_shared<ngraph::opset7::Result>(max_pool_operation);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                            ngraph::ParameterVector{input_params_convolution,
                                                                    input_params_add});
}

template <typename ActivationT, typename ... Args>
std::shared_ptr<ngraph::Function> createReferenceFunctionConvAddAct(Args&& ... args) {
    auto input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 3, 64, 64});

    auto input_params_add = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 3, 64, 64});

    auto weights = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{3, 3, 1, 1}, {1});
    auto bias = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{3, 1, 1}, {1});
    auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                                weights,
                                                                                ngraph::Strides{1, 1},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::CoordinateDiff{0, 0},
                                                                                ngraph::Strides{1, 1});

    auto add_operation = std::make_shared<ngraph::opset7::Add>(convolution_operation,
                                                                        input_params_convolution);

    auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(add_operation,
                                                                        ngraph::Strides{1, 1},
                                                                        ngraph::Shape{1, 1},
                                                                        ngraph::Shape{1, 1},
                                                                        ngraph::Shape{1, 1});

    auto activation = std::make_shared<ActivationT>(max_pool_operation, std::forward<Args>(args) ...);

    auto result = std::make_shared<ngraph::opset7::Result>(activation);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params_convolution,
                                                                            input_params_add});
}

TEST(TransformationTests, ReorderActivationAndPoolingTestConvAddActActivationRelu) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        func = createFunctionConvAddAct<ngraph::opset7::Relu>();

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createReferenceFunctionConvAddAct<ngraph::opset7::Relu>();

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, ReorderActivationAndPoolingTestConvAddActActivationSigmoid) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        func = createFunctionConvAddAct<ngraph::opset7::Sigmoid>();

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createReferenceFunctionConvAddAct<ngraph::opset7::Sigmoid>();

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, ReorderActivationAndPoolingTestConvAddActActivationTanh) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        func = createFunctionConvAddAct<ngraph::opset7::Tanh>();

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createReferenceFunctionConvAddAct<ngraph::opset7::Tanh>();

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, ReorderActivationAndPoolingTestConvAddActActivationAbs) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        func = createFunctionConvAddAct<ngraph::opset7::Abs>();

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createReferenceFunctionConvAddAct<ngraph::opset7::Abs>();

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, ReorderActivationAndPoolingTestConvAddActActivationLog) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        func = createFunctionConvAddAct<ngraph::opset7::Log>();

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createReferenceFunctionConvAddAct<ngraph::opset7::Log>();

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, ReorderActivationAndPoolingTestConvAddActActivationExp) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        func = createFunctionConvAddAct<ngraph::opset7::Exp>();

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createReferenceFunctionConvAddAct<ngraph::opset7::Exp>();

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, ReorderActivationAndPoolingTestConvAddActActivationSign) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        func = createFunctionConvAddAct<ngraph::opset7::Sign>();

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createReferenceFunctionConvAddAct<ngraph::opset7::Sign>();

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, ReorderActivationAndPoolingTestConvAddActActivationClamp) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        func = createFunctionConvAddAct<ngraph::opset7::Clamp>(0.1, 0.2);

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createReferenceFunctionConvAddAct<ngraph::opset7::Clamp>(0.1, 0.2);

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

// Variant Convolution -> FakeQuantize -> MaxPool : ConvFqMp

TEST(TransformationTests, ReorderActivationAndPoolingTestConvFqMp) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 3, 64, 64});

        auto weights = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{3, 3, 1, 1}, {1});
        auto bias = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{3, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                  weights,
                                                                  ngraph::Strides{1, 1},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::Strides{1, 1});

        auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        auto fake_quantize_op = std::make_shared<ngraph::opset7::FakeQuantize>(convolution_operation, input_low,
                                                                                input_high, output_low,
                                                                                output_high, 11);

        auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(fake_quantize_op,
                                                                                    ngraph::Strides{1, 1},
                                                                                    ngraph::Shape{1, 1},
                                                                                    ngraph::Shape{1, 1},
                                                                                    ngraph::Shape{1, 1});

        auto result = std::make_shared<ngraph::opset7::Result>(max_pool_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params_convolution});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 3, 64, 64});

        auto weights = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{3, 3, 1, 1}, {1});
        auto bias = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{3, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                  weights,
                                                                  ngraph::Strides{1, 1},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::Strides{1, 1});

        auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(convolution_operation,
                                                                                    ngraph::Strides{1, 1},
                                                                                    ngraph::Shape{1, 1},
                                                                                    ngraph::Shape{1, 1},
                                                                                    ngraph::Shape{1, 1});

        auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        auto fake_quantize_op = std::make_shared<ngraph::opset7::FakeQuantize>(max_pool_operation, input_low,
                                                                                input_high, output_low,
                                                                                output_high, 11);

        auto result = std::make_shared<ngraph::opset7::Result>(fake_quantize_op);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params_convolution});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

// Variant Convolution -> Add -> FakeQuantize -> MaxPool : ConvAddFqMp

TEST(TransformationTests, ReorderActivationAndPoolingTestConvAddFqMp) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 3, 64, 64});

        auto input_params_add = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 3, 64, 64});

        auto weights = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{3, 3, 1, 1}, {1});
        auto bias = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{3, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                  weights,
                                                                  ngraph::Strides{1, 1},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::Strides{1, 1});

        auto add_operation = std::make_shared<ngraph::opset7::Add>(convolution_operation,
                                                               input_params_add);

        auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        auto fake_quantize_op = std::make_shared<ngraph::opset7::FakeQuantize>(add_operation, input_low,
                                                                                input_high, output_low,
                                                                                output_high, 11);

        auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(fake_quantize_op,
                                                                                    ngraph::Strides{1, 1},
                                                                                    ngraph::Shape{1, 1},
                                                                                    ngraph::Shape{1, 1},
                                                                                    ngraph::Shape{1, 1});

        auto result = std::make_shared<ngraph::opset7::Result>(max_pool_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params_convolution, input_params_add});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 3, 64, 64});

        auto input_params_add = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 3, 64, 64});

        auto weights = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{3, 3, 1, 1}, {1});
        auto bias = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{3, 1, 1}, {1});
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                  weights,
                                                                  ngraph::Strides{1, 1},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::Strides{1, 1});

        auto add_operation = std::make_shared<ngraph::opset7::Add>(convolution_operation,
                                                               input_params_add);

        auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(add_operation,
                                                                                    ngraph::Strides{1, 1},
                                                                                    ngraph::Shape{1, 1},
                                                                                    ngraph::Shape{1, 1},
                                                                                    ngraph::Shape{1, 1});

        auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        auto fake_quantize_op = std::make_shared<ngraph::opset7::FakeQuantize>(max_pool_operation, input_low,
                                                                                input_high, output_low,
                                                                                output_high, 11);

        auto result = std::make_shared<ngraph::opset7::Result>(fake_quantize_op);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params_convolution, input_params_add});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

// Variant Convolution -> Activation -> MaxPool : ConvActMp

template <typename ActivationT, typename ... Args>
std::shared_ptr<ngraph::Function> createFunctionConvActMp(Args&& ... args) {
    auto input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 3, 64, 64});

    auto weights = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{3, 3, 1, 1}, {1});
    auto bias = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{3, 1, 1}, {1});
    auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                  weights,
                                                                  ngraph::Strides{1, 1},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::Strides{1, 1});

    auto activation = std::make_shared<ActivationT>(convolution_operation, std::forward<Args>(args) ...);

    auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(activation,
                                                                                    ngraph::Strides{1, 1},
                                                                                    ngraph::Shape{1, 1},
                                                                                    ngraph::Shape{1, 1},
                                                                                    ngraph::Shape{1, 1});

    auto result = std::make_shared<ngraph::opset7::Result>(max_pool_operation);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params_convolution});
}

template <typename ActivationT, typename ... Args>
std::shared_ptr<ngraph::Function> createReferenceFunctionConvActMp(Args&& ... args) {
    auto input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 3, 64, 64});

    auto weights = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{3, 3, 1, 1}, {1});
    auto bias = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{3, 1, 1}, {1});
    auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                  weights,
                                                                  ngraph::Strides{1, 1},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::Strides{1, 1});

    auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(convolution_operation,
                                                                                    ngraph::Strides{1, 1},
                                                                                    ngraph::Shape{1, 1},
                                                                                    ngraph::Shape{1, 1},
                                                                                    ngraph::Shape{1, 1});

    auto activation = std::make_shared<ActivationT>(max_pool_operation, std::forward<Args>(args) ...);

    auto result = std::make_shared<ngraph::opset7::Result>(activation);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params_convolution});
}

TEST(TransformationTests, ReorderActivationAndPoolingTestConvActMpActivationRelu) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        func = createFunctionConvActMp<ngraph::opset7::Relu>();

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createReferenceFunctionConvActMp<ngraph::opset7::Relu>();

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, ReorderActivationAndPoolingTestConvActMpActivationSigmoid) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        func = createFunctionConvActMp<ngraph::opset7::Sigmoid>();

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createReferenceFunctionConvActMp<ngraph::opset7::Sigmoid>();

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, ReorderActivationAndPoolingTestConvActMpActivationTanh) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        func = createFunctionConvActMp<ngraph::opset7::Tanh>();

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createReferenceFunctionConvActMp<ngraph::opset7::Tanh>();

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, ReorderActivationAndPoolingTestConvActMpActivationAbs) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        func = createFunctionConvActMp<ngraph::opset7::Abs>();

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createReferenceFunctionConvActMp<ngraph::opset7::Abs>();

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, ReorderActivationAndPoolingTestConvActMpActivationLog) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        func = createFunctionConvActMp<ngraph::opset7::Log>();

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createReferenceFunctionConvActMp<ngraph::opset7::Log>();

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, ReorderActivationAndPoolingTestConvActMpActivationExp) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        func = createFunctionConvActMp<ngraph::opset7::Exp>();

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createReferenceFunctionConvActMp<ngraph::opset7::Exp>();

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, ReorderActivationAndPoolingTestConvActMpActivationSign) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        func = createFunctionConvActMp<ngraph::opset7::Sign>();

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createReferenceFunctionConvActMp<ngraph::opset7::Sign>();

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, ReorderActivationAndPoolingTestConvActMpActivationClamp) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        func = createFunctionConvActMp<ngraph::opset7::Clamp>(0.1, 0.2);

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();

        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createReferenceFunctionConvActMp<ngraph::opset7::Clamp>(0.1, 0.2);

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

} // namespace testing
