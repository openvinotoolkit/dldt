//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "misc.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, add)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

    vector<float> a{1, 2, 3, 4};
    vector<float> b{5, 6, 7, 8};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {6, 8, 10, 12});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, add_overload)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A + B, ParameterVector{A, B});

    vector<float> a{1, 2, 3, 4};
    vector<float> b{5, 6, 7, 8};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {6, 8, 10, 12});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, add_in_place)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto T = A + B;
    auto T2 = T + T;
    auto T3 = T2 + T2;
    auto T4 = T3 + T3;

    auto f = make_shared<Function>(T4, ParameterVector{A, B});

    vector<float> a{1, 2, 3, 4};
    vector<float> b{5, 6, 7, 8};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {48, 64, 80, 96});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, add_large_tensors)
{
    Shape shape{10, 10, 10, 10, 10, 10, 10, 5, 5, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto T = make_shared<op::v1::Add>(A, B);

    auto f = make_shared<Function>(T, ParameterVector{A, B});

    vector<int32_t> a, b;
    a.reserve(shape_size(shape));
    b.reserve(shape_size(shape));

    std::cout << "Generating random input\n";
    {
        testing::internal::Random random(12345);
        for (size_t i = 0; i < shape_size(shape); ++i)
        {
            a.push_back(random.Generate(1000));
        }
    }

    std::cout << "Generating expected results\n";
    {
        for (const auto& x : a)
        {
            b.push_back(x * 2);
        }
    }

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>(a);
    test_case.add_input<int32_t>(a);
    test_case.add_expected_output<int32_t>(shape, b);

    std::cout << "Running the test\n";
    {
        test_case.run();
    }

    test_case.add_input<int32_t>(a);
    test_case.add_input<int32_t>(a);
    test_case.add_expected_output<int32_t>(shape, b);

    std::cout << "Running the test single threaded\n";
    {
        set_environment("REF_SINGLE_THREADED", "1", 1);
        test_case.run();
    }
}
