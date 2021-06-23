// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/unary_test.hpp"

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, elu)
{
    test_unary<TestEngine, element::f32>(
        unary_func<op::Elu>(0.5f),
        {-2.f, 3.f, -2.f, 1.f, -1.f, 0.f},
        {-0.432332358f, 3.f, -0.432332358f, 1.f, -0.316060279f, 0.f},
        Shape{3, 2});
}

NGRAPH_TEST(${BACKEND_NAME}, elu_negative_alpha)
{
    test_unary<TestEngine, element::f32>(unary_func<op::Elu>(-1.f),
                                         {-2.f, 3.f, -2.f, 1.f, -1.f, 0.f},
                                         {0.864664717f, 3.f, 0.864664717f, 1.f, 0.632120559f, 0.f},
                                         Shape{3, 2});
}