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

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include <iostream>
#include <numeric>
#include <string>

#include "gtest/gtest.h"
#include "runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

using Attrs = op::v6::ExperimentalDetectronPriorGridGenerator::Attributes;
using GridGenerator = op::v6::ExperimentalDetectronPriorGridGenerator;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, experimental_detectron_prior_grid_eval)
{
    std::vector<std::vector<float>> priors_value = {
        {-24.5, -12.5, 24.5, 12.5, -16.5, -16.5, 16.5, 16.5, -12.5, -24.5, 12.5, 24.5},
        {-44.5, -24.5, 44.5, 24.5, -32.5, -32.5, 32.5, 32.5, -24.5, -44.5, 24.5, 44.5},
        {-364.5, -184.5, 364.5, 184.5, -256.5, -256.5, 256.5, 256.5, -180.5, -360.5, 180.5, 360.5},
        {-180.5, -88.5, 180.5, 88.5, -128.5, -128.5, 128.5, 128.5, -92.5, -184.5, 92.5, 184.5}};

    struct ShapesAndAttrs
    {
        Attrs attrs;
        Shape priors_shape;
        Shape feature_map_shape;
        Shape im_data_shape;
        Shape ref_out_shape;
    };

    std::vector<ShapesAndAttrs> shapes_and_attrs = {
        {{true, 0, 0, 4.0f, 4.0f}, {3, 4}, {1, 16, 4, 5}, {1, 3, 100, 200}, {60, 4}},
        {{false, 0, 0, 8.0f, 8.0f}, {3, 4}, {1, 16, 3, 7}, {1, 3, 100, 200}, {3, 7, 3, 4}},
        {{true, 3, 6, 64.0f, 64.0f}, {3, 4}, {1, 16, 100, 100}, {1, 3, 100, 200}, {30000, 4}},
        {{false, 5, 3, 32.0f, 32.0f},
         {3, 4},
         {1, 16, 100, 100},
         {1, 3, 100, 200},
         {100, 100, 3, 4}}};

    std::vector<std::vector<float>> expected_results = {
        {-22.5, -10.5, 26.5,  14.5,  -14.5, -14.5, 18.5,  18.5,  -10.5, -22.5, 14.5,  26.5,  -18.5,
         -10.5, 30.5,  14.5,  -10.5, -14.5, 22.5,  18.5,  -6.5,  -22.5, 18.5,  26.5,  -14.5, -10.5,
         34.5,  14.5,  -6.5,  -14.5, 26.5,  18.5,  -2.5,  -22.5, 22.5,  26.5,  -10.5, -10.5, 38.5,
         14.5,  -2.5,  -14.5, 30.5,  18.5,  1.5,   -22.5, 26.5,  26.5,  -6.5,  -10.5, 42.5,  14.5,
         1.5,   -14.5, 34.5,  18.5,  5.5,   -22.5, 30.5,  26.5,  -22.5, -6.5,  26.5,  18.5,  -14.5,
         -10.5, 18.5,  22.5,  -10.5, -18.5, 14.5,  30.5,  -18.5, -6.5,  30.5,  18.5,  -10.5, -10.5,
         22.5,  22.5,  -6.5,  -18.5, 18.5,  30.5,  -14.5, -6.5,  34.5,  18.5,  -6.5,  -10.5, 26.5,
         22.5,  -2.5,  -18.5, 22.5,  30.5,  -10.5, -6.5,  38.5,  18.5,  -2.5,  -10.5, 30.5,  22.5,
         1.5,   -18.5, 26.5,  30.5,  -6.5,  -6.5,  42.5,  18.5,  1.5,   -10.5, 34.5,  22.5,  5.5,
         -18.5, 30.5,  30.5,  -22.5, -2.5,  26.5,  22.5,  -14.5, -6.5,  18.5,  26.5,  -10.5, -14.5,
         14.5,  34.5,  -18.5, -2.5,  30.5,  22.5,  -10.5, -6.5,  22.5,  26.5,  -6.5,  -14.5, 18.5,
         34.5,  -14.5, -2.5,  34.5,  22.5,  -6.5,  -6.5,  26.5,  26.5,  -2.5,  -14.5, 22.5,  34.5,
         -10.5, -2.5,  38.5,  22.5,  -2.5,  -6.5,  30.5,  26.5,  1.5,   -14.5, 26.5,  34.5,  -6.5,
         -2.5,  42.5,  22.5,  1.5,   -6.5,  34.5,  26.5,  5.5,   -14.5, 30.5,  34.5,  -22.5, 1.5,
         26.5,  26.5,  -14.5, -2.5,  18.5,  30.5,  -10.5, -10.5, 14.5,  38.5,  -18.5, 1.5,   30.5,
         26.5,  -10.5, -2.5,  22.5,  30.5,  -6.5,  -10.5, 18.5,  38.5,  -14.5, 1.5,   34.5,  26.5,
         -6.5,  -2.5,  26.5,  30.5,  -2.5,  -10.5, 22.5,  38.5,  -10.5, 1.5,   38.5,  26.5,  -2.5,
         -2.5,  30.5,  30.5,  1.5,   -10.5, 26.5,  38.5,  -6.5,  1.5,   42.5,  26.5,  1.5,   -2.5,
         34.5,  30.5,  5.5,   -10.5, 30.5,  38.5},
        {-40.5, -20.5, 48.5,  28.5,  -28.5, -28.5, 36.5,  36.5,  -20.5, -40.5, 28.5,  48.5,  -32.5,
         -20.5, 56.5,  28.5,  -20.5, -28.5, 44.5,  36.5,  -12.5, -40.5, 36.5,  48.5,  -24.5, -20.5,
         64.5,  28.5,  -12.5, -28.5, 52.5,  36.5,  -4.5,  -40.5, 44.5,  48.5,  -16.5, -20.5, 72.5,
         28.5,  -4.5,  -28.5, 60.5,  36.5,  3.5,   -40.5, 52.5,  48.5,  -8.5,  -20.5, 80.5,  28.5,
         3.5,   -28.5, 68.5,  36.5,  11.5,  -40.5, 60.5,  48.5,  -0.5,  -20.5, 88.5,  28.5,  11.5,
         -28.5, 76.5,  36.5,  19.5,  -40.5, 68.5,  48.5,  7.5,   -20.5, 96.5,  28.5,  19.5,  -28.5,
         84.5,  36.5,  27.5,  -40.5, 76.5,  48.5,  -40.5, -12.5, 48.5,  36.5,  -28.5, -20.5, 36.5,
         44.5,  -20.5, -32.5, 28.5,  56.5,  -32.5, -12.5, 56.5,  36.5,  -20.5, -20.5, 44.5,  44.5,
         -12.5, -32.5, 36.5,  56.5,  -24.5, -12.5, 64.5,  36.5,  -12.5, -20.5, 52.5,  44.5,  -4.5,
         -32.5, 44.5,  56.5,  -16.5, -12.5, 72.5,  36.5,  -4.5,  -20.5, 60.5,  44.5,  3.5,   -32.5,
         52.5,  56.5,  -8.5,  -12.5, 80.5,  36.5,  3.5,   -20.5, 68.5,  44.5,  11.5,  -32.5, 60.5,
         56.5,  -0.5,  -12.5, 88.5,  36.5,  11.5,  -20.5, 76.5,  44.5,  19.5,  -32.5, 68.5,  56.5,
         7.5,   -12.5, 96.5,  36.5,  19.5,  -20.5, 84.5,  44.5,  27.5,  -32.5, 76.5,  56.5,  -40.5,
         -4.5,  48.5,  44.5,  -28.5, -12.5, 36.5,  52.5,  -20.5, -24.5, 28.5,  64.5,  -32.5, -4.5,
         56.5,  44.5,  -20.5, -12.5, 44.5,  52.5,  -12.5, -24.5, 36.5,  64.5,  -24.5, -4.5,  64.5,
         44.5,  -12.5, -12.5, 52.5,  52.5,  -4.5,  -24.5, 44.5,  64.5,  -16.5, -4.5,  72.5,  44.5,
         -4.5,  -12.5, 60.5,  52.5,  3.5,   -24.5, 52.5,  64.5,  -8.5,  -4.5,  80.5,  44.5,  3.5,
         -12.5, 68.5,  52.5,  11.5,  -24.5, 60.5,  64.5,  -0.5,  -4.5,  88.5,  44.5,  11.5,  -12.5,
         76.5,  52.5,  19.5,  -24.5, 68.5,  64.5,  7.5,   -4.5,  96.5,  44.5,  19.5,  -12.5, 84.5,
         52.5,  27.5,  -24.5, 76.5,  64.5},
        {-332.5, -152.5, 396.5, 216.5, -224.5, -224.5, 288.5, 288.5, -148.5, -328.5, 212.5, 392.5,
         -268.5, -152.5, 460.5, 216.5, -160.5, -224.5, 352.5, 288.5, -84.5,  -328.5, 276.5, 392.5,
         -204.5, -152.5, 524.5, 216.5, -96.5,  -224.5, 416.5, 288.5, -20.5,  -328.5, 340.5, 392.5,
         -140.5, -152.5, 588.5, 216.5, -32.5,  -224.5, 480.5, 288.5, 43.5,   -328.5, 404.5, 392.5,
         -76.5,  -152.5, 652.5, 216.5, 31.5,   -224.5, 544.5, 288.5, 107.5,  -328.5, 468.5, 392.5,
         -12.5,  -152.5, 716.5, 216.5, 95.5,   -224.5, 608.5, 288.5, 171.5,  -328.5, 532.5, 392.5,
         -332.5, -88.5,  396.5, 280.5, -224.5, -160.5, 288.5, 352.5, -148.5, -264.5, 212.5, 456.5,
         -268.5, -88.5,  460.5, 280.5, -160.5, -160.5, 352.5, 352.5, -84.5,  -264.5, 276.5, 456.5,
         -204.5, -88.5,  524.5, 280.5, -96.5,  -160.5, 416.5, 352.5, -20.5,  -264.5, 340.5, 456.5,
         -140.5, -88.5,  588.5, 280.5, -32.5,  -160.5, 480.5, 352.5, 43.5,   -264.5, 404.5, 456.5,
         -76.5,  -88.5,  652.5, 280.5, 31.5,   -160.5, 544.5, 352.5, 107.5,  -264.5, 468.5, 456.5,
         -12.5,  -88.5,  716.5, 280.5, 95.5,   -160.5, 608.5, 352.5, 171.5,  -264.5, 532.5, 456.5,
         -332.5, -24.5,  396.5, 344.5, -224.5, -96.5,  288.5, 416.5, -148.5, -200.5, 212.5, 520.5,
         -268.5, -24.5,  460.5, 344.5, -160.5, -96.5,  352.5, 416.5, -84.5,  -200.5, 276.5, 520.5,
         -204.5, -24.5,  524.5, 344.5, -96.5,  -96.5,  416.5, 416.5, -20.5,  -200.5, 340.5, 520.5,
         -140.5, -24.5,  588.5, 344.5, -32.5,  -96.5,  480.5, 416.5, 43.5,   -200.5, 404.5, 520.5,
         -76.5,  -24.5,  652.5, 344.5, 31.5,   -96.5,  544.5, 416.5, 107.5,  -200.5, 468.5, 520.5,
         -12.5,  -24.5,  716.5, 344.5, 95.5,   -96.5,  608.5, 416.5, 171.5,  -200.5, 532.5, 520.5},
        {-164.5, -72.5, 196.5, 104.5, -112.5, -112.5, 144.5, 144.5, -76.5, -168.5, 108.5, 200.5,
         -132.5, -72.5, 228.5, 104.5, -80.5,  -112.5, 176.5, 144.5, -44.5, -168.5, 140.5, 200.5,
         -100.5, -72.5, 260.5, 104.5, -48.5,  -112.5, 208.5, 144.5, -12.5, -168.5, 172.5, 200.5,
         -164.5, -40.5, 196.5, 136.5, -112.5, -80.5,  144.5, 176.5, -76.5, -136.5, 108.5, 232.5,
         -132.5, -40.5, 228.5, 136.5, -80.5,  -80.5,  176.5, 176.5, -44.5, -136.5, 140.5, 232.5,
         -100.5, -40.5, 260.5, 136.5, -48.5,  -80.5,  208.5, 176.5, -12.5, -136.5, 172.5, 232.5,
         -164.5, -8.5,  196.5, 168.5, -112.5, -48.5,  144.5, 208.5, -76.5, -104.5, 108.5, 264.5,
         -132.5, -8.5,  228.5, 168.5, -80.5,  -48.5,  176.5, 208.5, -44.5, -104.5, 140.5, 264.5,
         -100.5, -8.5,  260.5, 168.5, -48.5,  -48.5,  208.5, 208.5, -12.5, -104.5, 172.5, 264.5,
         -164.5, 23.5,  196.5, 200.5, -112.5, -16.5,  144.5, 240.5, -76.5, -72.5,  108.5, 296.5,
         -132.5, 23.5,  228.5, 200.5, -80.5,  -16.5,  176.5, 240.5, -44.5, -72.5,  140.5, 296.5,
         -100.5, 23.5,  260.5, 200.5, -48.5,  -16.5,  208.5, 240.5, -12.5, -72.5,  172.5, 296.5,
         -164.5, 55.5,  196.5, 232.5, -112.5, 15.5,   144.5, 272.5, -76.5, -40.5,  108.5, 328.5,
         -132.5, 55.5,  228.5, 232.5, -80.5,  15.5,   176.5, 272.5, -44.5, -40.5,  140.5, 328.5,
         -100.5, 55.5,  260.5, 232.5, -48.5,  15.5,   208.5, 272.5, -12.5, -40.5,  172.5, 328.5}};

    std::size_t i = 0;
    for (const auto& s : shapes_and_attrs)
    {
        auto priors = std::make_shared<op::Parameter>(element::f32, s.priors_shape);
        auto feature_map = std::make_shared<op::Parameter>(element::f32, s.feature_map_shape);
        auto im_data = std::make_shared<op::Parameter>(element::f32, s.im_data_shape);

        auto grid_gen = std::make_shared<GridGenerator>(priors, feature_map, im_data, s.attrs);

        auto f = make_shared<Function>(grid_gen, ParameterVector{priors, feature_map, im_data});

        auto backend = runtime::Backend::create("${BACKEND_NAME}");

        auto priors_data = priors_value[i];

        auto& ref_results = expected_results[i];

        std::vector<float> feature_map_data(shape_size(s.feature_map_shape));
        std::iota(feature_map_data.begin(), feature_map_data.end(), 0);
        std::vector<float> image_data(shape_size(s.im_data_shape));
        std::iota(image_data.begin(), image_data.end(), 0);

        auto output_priors = backend->create_tensor(element::f32, s.ref_out_shape);

        auto backend_priors = backend->create_tensor(element::f32, s.priors_shape);
        auto backend_feature_map = backend->create_tensor(element::f32, s.feature_map_shape);
        auto backend_im_data = backend->create_tensor(element::f32, s.im_data_shape);
        copy_data(backend_priors, priors_data);
        copy_data(backend_feature_map, feature_map_data);
        copy_data(backend_im_data, image_data);

        auto handle = backend->compile(f);

        handle->call({output_priors}, {backend_priors, backend_feature_map, backend_im_data});

        auto output_priors_value = read_vector<float>(output_priors);

        std::cout << std::string(80, '*') << "\n";
        std::cout << "Actual number of floats in calculated result: "
                  << output_priors_value.size() << "\n";
        auto num_of_expected_results = expected_results[i].size();
        std::cout << "Actual calculated result (first "
                  << num_of_expected_results << " floats):\n    ";

        for (size_t j = 0; j < num_of_expected_results; ++j)
        {
            std::cout << output_priors_value[j] << ", ";
        }
        std::cout << "\n";

        std::vector<float> actual_results(output_priors_value.begin(),
                                          output_priors_value.begin() + ref_results.size());
        EXPECT_EQ(ref_results, actual_results);
        ++i;
    }
}
