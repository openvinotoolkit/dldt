//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif
#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "core/null_node.hpp"
#include "gtest/gtest.h"
#include "onnx_import/onnx.hpp"
#include "onnx_import/onnx_utils.hpp"
#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, onnx_prior_box)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/prior_box.prototxt"));
    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);
    std::vector<float> A(3 * 2 * 2);
    std::vector<float> B(3 * 6 * 6);
    std::vector<float> output = {
        -2.3200002, -2.3200002,  3.6533334,  3.6533334,   -3.7053659,  -3.7053659, 5.0386992,
        5.0386992,  -0.98666668, -2.3200002, 4.9866667,   3.6533334,   -2.3720326, -3.7053659,
        6.3720322,  5.0386992,   -2.3200002, -0.98666668, 3.6533334,   4.9866667,  -3.7053659,
        -2.3720326, 5.0386992,   6.3720322,  -0.98666668, -0.98666668, 4.9866667,  4.9866667,
        -2.3720326, -2.3720326,  6.3720322,  6.3720322,   0.1,         0.1,        0.2,
        0.2,        0.1,         0.1,        0.2,         0.2,         0.1,        0.1,
        0.2,        0.2,         0.1,        0.1,         0.2,         0.2,        0.1,
        0.1,        0.2,         0.2,        0.1,         0.1,         0.2,        0.2,
        0.1,        0.1,         0.2,        0.2,         0.1,         0.1,        0.2,
        0.2,
    };
    test_case.add_input<float>(A);
    test_case.add_input<float>(B);
    test_case.add_expected_output<float>(Shape{1, 2, 32}, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_detection_output)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/detection_output.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    auto gen_vector = [](size_t size, float min, float max) -> std::vector<float> {
        float step = (max - min) / size;
        float next = min - step;

        std::vector<float> out(size);
        std::generate(out.begin(), out.end(), [&next, &step] { return next += step; });
        return out;
    };

    std::vector<float> logits = gen_vector(12, -2, 2);
    std::vector<float> class_preds = gen_vector(9, 0, 1);
    std::vector<float> proposals = gen_vector(12 * 2, 0, 1);
    std::vector<float> output = {0, 1, 0.777778, 0.279849,   0.283779,   0.562743,   0.695387,
                                 0, 1, 0.444444, 0.12963,    0.176075,   0.212963,   0.284573,
                                 0, 2, 0.888889, 0.279849,   0.283779,   0.562743,   0.695387,
                                 0, 2, 0.555556, 0.12963,    0.176075,   0.212963,   0.284573,
                                 0, 2, 0.222222, -0.0608094, -0.0142007, -0.0225239, 0.0304044};
    test_case.add_input<float>(logits);
    test_case.add_input<float>(class_preds);
    test_case.add_input<float>(proposals);
    test_case.add_expected_output<float>(Shape{1, 1, 5, 7}, output);
    int tolerance_bits = 6;
    test_case.run(tolerance_bits);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_group_norm)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/group_norm.prototxt"));
    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);
    Shape shape{2, 8, 2, 2};
    int size = shape_size(shape);
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0);
    std::vector<float> output = {
        -0.52752507, -0.09108937, 0.3453464, 0.78178215, 2.4364357, 3.309307,  4.1821785, 5.05505,
        -1.5825753,  -0.27326822, 1.0360391, 2.3453465,  4.8728714, 6.618614,  8.364357,  10.1101,
        -2.6376252,  -0.45544672, 1.726732,  3.9089108,  7.309307,  9.927921,  12.546536, 15.165151,
        -3.6926756,  -0.6376257,  2.4174247, 5.472475,   9.745743,  13.237228, 16.728714, 20.2202,
        -0.52752507, -0.09108937, 0.3453464, 0.78178215, 2.4364357, 3.309307,  4.1821785, 5.05505,
        -1.5825753,  -0.27326822, 1.0360391, 2.3453465,  4.8728714, 6.618614,  8.364357,  10.1101,
        -2.6376252,  -0.45544672, 1.726732,  3.9089108,  7.309307,  9.927921,  12.546536, 15.165151,
        -3.6926756,  -0.6376257,  2.4174247, 5.472475,   9.745743,  13.237228, 16.728714, 20.2202,
    };

    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(shape, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_normalize)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/normalize.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 1);
    std::vector<float> output = {
        0.19334731,
        0.33806169,
        0.44846106,
        0.53452247,
        1.4501048,
        1.5212777,
        1.5696137,
        1.6035674,
        3.4802516,
        3.3806169,
        3.2887144,
        3.2071347,
    };
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{1, 3, 2, 2}, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_swish_with_beta)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/swish_with_beta.prototxt"));

    const Shape expected_output_shape{3};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{-0.5f, 0, 0.5f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {-0.2036667, 0.0, 0.2963333});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_swish_without_beta)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/swish_without_beta.prototxt"));

    const Shape expected_output_shape{3};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{-0.5f, 0, 0.5f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {-0.18877034, 0.0, 0.31122968});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}
