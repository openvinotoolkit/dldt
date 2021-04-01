// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, gather_4d_indices_axis_0_uint8)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2, 3, 4};
    Shape out_shape{2, 2, 3, 4, 2};
    auto P = make_shared<op::Parameter>(element::u8, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<uint8_t>({10, 11, 20, 21, 30, 31});
    test_case.add_input<int32_t>({0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,
                                  0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,
                                  0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2});
    test_case.add_expected_output<uint8_t>(
        out_shape, {10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21,
                    20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31,
                    10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21,
                    20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31,
                    10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gather_4d_indices_axis_0_2d_input)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2, 3, 4};
    Shape out_shape{2, 2, 3, 4, 2};
    auto P = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto test_case = test::TestCase<TestEngine>(f);

    // clang-format off
    test_case.add_input<float>({1.0f, 1.1f,
                                2.0f, 2.1f,
                                3.0f, 3.1f});

    test_case.add_input<int32_t>({0, 1, 1, 2,
                                  0, 1, 1, 2,
                                  0, 1, 1, 2,

                                  0, 1, 1, 2,
                                  0, 1, 1, 2,
                                  0, 1, 1, 2,


                                  0, 1, 1, 2,
                                  0, 1, 1, 2,
                                  0, 1, 1, 2,

                                  0, 1, 1, 2,
                                  0, 1, 1, 2,
                                  0, 1, 1, 2});
    test_case.add_expected_output<float>(
        out_shape,
        { 1.0f, 1.1f,
          2.0f, 2.1f,
          2.0f, 2.1f,
          3.0f, 3.1f,

          1.0f, 1.1f,
          2.0f, 2.1f,
          2.0f, 2.1f,
          3.0f, 3.1f,

          1.0f, 1.1f,
          2.0f, 2.1f,
          2.0f, 2.1f,
          3.0f, 3.1f,


          1.0f, 1.1f,
          2.0f, 2.1f,
          2.0f, 2.1f,
          3.0f, 3.1f,

          1.0f, 1.1f,
          2.0f, 2.1f,
          2.0f, 2.1f,
          3.0f, 3.1f,

          1.0f, 1.1f,
          2.0f, 2.1f,
          2.0f, 2.1f,
          3.0f, 3.1f,



          1.0f, 1.1f,
          2.0f, 2.1f,
          2.0f, 2.1f,
          3.0f, 3.1f,


          1.0f, 1.1f,
          2.0f, 2.1f,
          2.0f, 2.1f,
          3.0f, 3.1f,

          1.0f, 1.1f,
          2.0f, 2.1f,
          2.0f, 2.1f,
          3.0f, 3.1f,


          1.0f, 1.1f,
          2.0f, 2.1f,
          2.0f, 2.1f,
          3.0f, 3.1f,

          1.0f, 1.1f,
          2.0f, 2.1f,
          2.0f, 2.1f,
          3.0f, 3.1f,

          1.0f, 1.1f,
          2.0f, 2.1f,
          2.0f, 2.1f,
          3.0f, 3.1f});
    // clang-format on
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_3d_indices_axis_0_2d_input)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 3, 4};
    Shape out_shape{2, 3, 4, 2};
    auto P = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto test_case = test::TestCase<TestEngine>(f);
    // clang-format off
    test_case.add_input<float>({1.0f, 1.1f,
                                2.0f, 2.1f,
                                3.0f, 3.1f});
    test_case.add_input<int32_t>(
        {0, 1, 1, 2,
         0, 1, 1, 2,
         0, 1, 1, 2,

         0, 1, 1, 2,
         0, 1, 1, 2,
         0, 1, 1, 2});
    test_case.add_expected_output<float>(
        out_shape, {1.0f, 1.1f,
                    2.0f, 2.1f,
                    2.0f, 2.1f,
                    3.0f, 3.1f,

                    1.0f, 1.1f,
                    2.0f, 2.1f,
                    2.0f, 2.1f,
                    3.0f, 3.1f,

                    1.0f, 1.1f,
                    2.0f, 2.1f,
                    2.0f, 2.1f,
                    3.0f, 3.1f,


                    1.0f, 1.1f,
                    2.0f, 2.1f,
                    2.0f, 2.1f,
                    3.0f, 3.1f,

                    1.0f, 1.1f,
                    2.0f, 2.1f,
                    2.0f, 2.1f,
                    3.0f, 3.1f,

                    1.0f, 1.1f,
                    2.0f, 2.1f,
                    2.0f, 2.1f,
                    3.0f, 3.1f});
    // clang-format on
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_2d_indices_axis_0_2d_input)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto test_case = test::TestCase<TestEngine>(f);
    // clang-format off
    test_case.add_input<float>({1.0f, 1.1f,
                                2.0f, 2.1f,
                                3.0f, 3.1f});
    // clang-format on
    test_case.add_input<int32_t>({0, 1, 1, 2});
    // clang-format off
    test_case.add_expected_output<float>(out_shape,
                                         {1.0f, 1.1f,
                                          2.0f, 2.1f,

                                          2.0f, 2.1f,
                                          3.0f, 3.1f});
    // clang-format on
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_2d_negative_and_positive_indices_axis_0_2d_input)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto test_case = test::TestCase<TestEngine>(f);

    // clang-format off
    test_case.add_input<float>({1.0f, 1.1f,
                                2.0f, 2.1f,
                                3.0f, 3.1f});
    // clang-format on

    test_case.add_input<int32_t>({0, -2, 1, 2});

    // clang-format off
    test_case.add_expected_output<float>(out_shape,
                                         {1.0f, 1.1f,
                                          2.0f, 2.1f,

                                          2.0f, 2.1f,
                                          3.0f, 3.1f});
    // clang-format on

    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_1d_indices_axis_0_1d_input)
{
    Shape data_shape{3};
    Shape indices_shape{2};
    Shape out_shape{2};
    auto P = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<int32_t>({1, 0});
    test_case.add_expected_output<float>(out_shape, {2.0f, 1.0f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_scalar_indices_axis_0_2d_input)
{
    Shape data_shape{3, 2};
    Shape indices_shape{};
    Shape out_shape{2};
    auto P = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f});
    test_case.add_input<int32_t>({1});
    test_case.add_expected_output<float>(out_shape, {2.0f, 2.1f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_2d_indices_axis_1_2d_input)
{
    Shape data_shape{3, 3};
    Shape indices_shape{1, 2};
    Shape out_shape{3, 1, 2};
    auto P = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {1});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto test_case = test::TestCase<TestEngine>(f);

    // clang-format off
    test_case.add_input<float>({1.0f, 1.1f, 1.2f,
                                2.0f, 2.1f, 2.2f,
                                3.0f, 3.1f, 3.2f});
    // clang-format on
    test_case.add_input<int32_t>({0, 2});

    // clang-format off
    test_case.add_expected_output<float>(out_shape, {1.0f, 1.2f,
                                                     2.0f, 2.2f,
                                                     3.0f, 3.2f});
    // clang-format on
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_1d_indices_axis_2_4d_input)
{
    Shape data_shape{2, 2, 3, 3};
    Shape indices_shape{2};
    Shape out_shape{2, 2, 2, 3};
    auto P = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {2});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto test_case = test::TestCase<TestEngine>(f);
    // clang-format off
    test_case.add_input<float>({  1.0f,   1.1f,   1.2f,
                                  2.0f,   2.1f,   2.2f,
                                  3.0f,   3.1f,   3.2f,

                                 11.0f,  11.1f,  11.2f,
                                 12.0f,  12.1f,  12.2f,
                                 13.0f,  13.1f,  13.2f,


                                101.0f, 101.1f, 101.2f,
                                102.0f, 102.1f, 102.2f,
                                103.0f, 103.1f, 103.2f,

                                111.0f, 111.1f, 111.2f,
                                112.0f, 112.1f, 112.2f,
                                113.0f, 113.1f, 113.2f});
    // clang-format on
    test_case.add_input<int32_t>({0, 2});
    // clang-format off
    test_case.add_expected_output<float>(
        out_shape, {  1.0f,   1.1f,   1.2f,
                      3.0f,   3.1f,   3.2f,

                     11.0f,  11.1f,  11.2f,
                     13.0f,  13.1f,  13.2f,


                    101.0f, 101.1f, 101.2f,
                    103.0f, 103.1f, 103.2f,

                    111.0f, 111.1f, 111.2f,
                    113.0f, 113.1f, 113.2f});
    // clang-format on
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_scalar_indices_axis_1_2d_input)
{
    Shape data_shape{3, 3};
    Shape indices_shape{};
    Shape out_shape{3};
    auto P = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {1});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f, 3.0f, 3.1f, 3.2f});
    test_case.add_input<int32_t>({0});
    test_case.add_expected_output<float>(out_shape, {1.0f, 2.0f, 3.0f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_axis_0_int8)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::i8, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int8_t>({10, 11, 20, 21, 30, 31});
    test_case.add_input<int32_t>({0, 1, 1, 2});
    test_case.add_expected_output<int8_t>(out_shape, {10, 11, 20, 21, 20, 21, 30, 31});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_axis_0_int16)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::i16, data_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int16_t>({10, 11, 20, 21, 30, 31});
    test_case.add_input<int64_t>({0, 1, 1, 2});
    test_case.add_expected_output<int16_t>(out_shape, {10, 11, 20, 21, 20, 21, 30, 31});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_axis_0_int32)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::i32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});
    // clang-format off
    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>({10, 11,
                                  20, 21,
                                  30, 31});
    test_case.add_input<int32_t>({0, 1,
                                  1, 2});
    test_case.add_expected_output<int32_t>(out_shape, {10, 11,
                                                       20, 21,

                                                       20, 21,
                                                       30, 31});
    // clang-format on
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_axis_0_int64)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::i64, data_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int64_t>({10, 11, 20, 21, 30, 31});
    test_case.add_input<int64_t>({0, 1, 1, 2});
    test_case.add_expected_output<int64_t>(out_shape, {10, 11, 20, 21, 20, 21, 30, 31});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_axis_0_uint8)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::u8, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<uint8_t>({10, 11, 20, 21, 30, 31});
    test_case.add_input<int32_t>({0, 1, 1, 2});
    test_case.add_expected_output<uint8_t>(out_shape, {10, 11, 20, 21, 20, 21, 30, 31});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_axis_0_uint16)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::u16, data_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<uint16_t>({10, 11, 20, 21, 30, 31});
    test_case.add_input<int64_t>({0, 1, 1, 2});
    test_case.add_expected_output<uint16_t>(out_shape, {10, 11, 20, 21, 20, 21, 30, 31});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_axis_0_uint32)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::u32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<uint32_t>({10, 11, 20, 21, 30, 31});
    test_case.add_input<int32_t>({0, 1, 1, 2});
    test_case.add_expected_output<uint32_t>(out_shape, {10, 11, 20, 21, 20, 21, 30, 31});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_axis_0_uint64)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::u64, data_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<uint64_t>({10, 11, 20, 21, 30, 31});
    test_case.add_input<int64_t>({0, 1, 1, 2});
    test_case.add_expected_output<uint64_t>(out_shape, {10, 11, 20, 21, 20, 21, 30, 31});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, gather_axis_0_bool)
{
    Shape data_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::boolean, data_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<char>({1, 1, 1, 0, 0, 1});
    test_case.add_input<int64_t>({0, 1, 1, 2});
    test_case.add_expected_output<char>(out_shape, {1, 1, 1, 0, 1, 0, 0, 1});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

//NGRAPH_TEST(${BACKEND_NAME}, gather_7_axis_0_bool)
//{
//    Shape data_shape{3, 2};
//    Shape indices_shape{2, 2};
//    Shape out_shape{2, 2, 2};
//    int64_t batch_dims = 1;
//
//    auto P = make_shared<op::Parameter>(element::boolean, data_shape);
//    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
//    auto A = op::Constant::create(element::i64, Shape{}, {1});
//    auto G = make_shared<op::v7::Gather>(P, I, A, batch_dims);
//    auto f = make_shared<Function>(G, ParameterVector{P, I});
//
//    auto test_case = test::TestCase<TestEngine>(f);
//    test_case.add_input<char>({1, 1, 1, 0, 0, 1});
//    test_case.add_input<int64_t>({0, 1, 1, 2});
//    test_case.add_expected_output<char>(out_shape, {1, 1, 1, 0, 1, 0, 0, 1});
//    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
//}
//
//NGRAPH_TEST(${BACKEND_NAME}, gather_7_3d_indices_axis_1_batch_dims_1)
//{
//    Shape data_shape{2, 5, 2};
//    Shape indices_shape{2, 2, 3};
//    Shape out_shape{2, 2, 3, 2};
//    auto P = make_shared<op::Parameter>(element::f32, data_shape);
//    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
//    auto A = op::Constant::create(element::i64, Shape{}, {1});
//    int64_t batch_dims = 1;
//    auto G = make_shared<op::v7::Gather>(P, I, A, batch_dims);
//    auto f = make_shared<Function>(G, ParameterVector{P, I});
//
//    auto test_case = test::TestCase<TestEngine>(f);
//
//    // clang-format off
//    test_case.add_input<float>({1.0f, 2.0f,
//                                3.0f, 4.0f,
//                                5.0f, 6.0f,
//                                7.0f, 8.0f,
//                                9.0f, 10.0f,
//
//                                11.0f, 12.0f,
//                                13.0f, 14.0f,
//                                15.0f, 16.0f,
//                                17.0f, 18.0f,
//                                19.0f, 20.0f});
//
//    test_case.add_input<int32_t>({0, 0, 4,
//                                  4, 0, 0,
//
//                                  1, 2, 4,
//                                  4, 3, 2});
//    test_case.add_expected_output<float>({1.0f, 2.0f,
//                                          1.0f, 2.0f,
//                                          9.0f, 10.0f,
//
//                                          9.0f, 10.0f,
//                                          1.0f, 2.0f,
//                                          1.0f, 2.0f,
//
//
//                                          13.0f, 14.0f,
//                                          15.0f, 16.0f,
//                                          19.0f, 20.0f,
//
//                                          19.0f, 20.0f,
//                                          17.0f, 18.0f,
//                                          15.0f, 16.0f});
//    // clang-format on
//    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
//}
