// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "common_test_utils/ngraph_test_utils.hpp"

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/convert_to_unsigned_nms_gather.hpp>
#include <transformations/init_node_info.hpp>

using namespace testing;
using namespace ngraph;
using namespace std;

TEST(TransformationTests, test_convert_to_unsigned_nms_gather_1) {
    // if Convert doesn't exist
    shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = make_shared<opset8::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = make_shared<opset8::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = make_shared<opset8::NonMaxSuppression>(boxes, scores);

        auto begin = opset8::Constant::create(element::i32, Shape{1}, {3});
        auto end = opset8::Constant::create(element::i32, Shape{1}, {4});
        auto strides = opset8::Constant::create(element::i32, Shape{1}, {1});
        auto ss_node = make_shared<opset8::StridedSlice>(nms->output(0), begin, end, strides, vector<int64_t>{1, 0}, vector<int64_t>{1, 0});

        // squeeze can be present as reshape
        auto squeeze_node = make_shared<opset8::Reshape>(ss_node, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        // usually input to gather data goes after reshape NMS scores
        auto reshape_node = make_shared<opset8::Reshape>(scores, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        auto gather = make_shared<opset8::Gather>(reshape_node, squeeze_node, opset8::Constant::create(element::i32, Shape{1}, {0}));

        f = make_shared<Function>(NodeVector{gather}, ParameterVector{boxes, scores});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertToUnsignedNmsGather>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto boxes = make_shared<opset8::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = make_shared<opset8::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = make_shared<opset8::NonMaxSuppression>(boxes, scores);

        auto begin = opset8::Constant::create(element::i32, Shape{1}, {3});
        auto end = opset8::Constant::create(element::i32, Shape{1}, {4});
        auto strides = opset8::Constant::create(element::i32, Shape{1}, {1});
        auto ss_node = make_shared<opset8::StridedSlice>(nms->output(0), begin, end, strides, vector<int64_t>{1, 0}, vector<int64_t>{1, 0});

        // squeeze can be present as reshape
        auto squeeze_node = make_shared<opset8::Reshape>(ss_node, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        auto convert = make_shared<opset8::Convert>(squeeze_node, element::Type_t::u64);
        auto reshape_node = make_shared<opset8::Reshape>(scores, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        auto gather = make_shared<opset8::Gather>(reshape_node, convert, opset8::Constant::create(element::i32, Shape{1}, {0}));

        f_ref = make_shared<Function>(NodeVector{gather}, ParameterVector{boxes, scores});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, test_convert_to_unsigned_nms_gather_2) {
    // if Convert already exists
    shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = make_shared<opset8::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = make_shared<opset8::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = make_shared<opset8::NonMaxSuppression>(boxes, scores);

        auto begin = opset8::Constant::create(element::i32, Shape{1}, {3});
        auto end = opset8::Constant::create(element::i32, Shape{1}, {4});
        auto strides = opset8::Constant::create(element::i32, Shape{1}, {1});
        auto ss_node = make_shared<opset8::StridedSlice>(nms->output(0), begin, end, strides, vector<int64_t>{1, 0}, vector<int64_t>{1, 0});

        // squeeze can be present as reshape
        auto squeeze_node = make_shared<opset8::Reshape>(ss_node, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        auto convert = make_shared<opset8::Convert>(squeeze_node, element::Type_t::i32);
        // usually input to gather data goes after reshape NMS scores
        auto reshape_node = make_shared<opset8::Reshape>(scores, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        auto gather = make_shared<opset8::Gather>(reshape_node, convert, opset8::Constant::create(element::i32, Shape{1}, {0}));

        f = make_shared<Function>(NodeVector{gather}, ParameterVector{boxes, scores});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertToUnsignedNmsGather>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto boxes = make_shared<opset8::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = make_shared<opset8::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = make_shared<opset8::NonMaxSuppression>(boxes, scores);

        auto begin = opset8::Constant::create(element::i32, Shape{1}, {3});
        auto end = opset8::Constant::create(element::i32, Shape{1}, {4});
        auto strides = opset8::Constant::create(element::i32, Shape{1}, {1});
        auto ss_node = make_shared<opset8::StridedSlice>(nms->output(0), begin, end, strides, vector<int64_t>{1, 0}, vector<int64_t>{1, 0});

        // squeeze can be present as reshape
        auto squeeze_node = make_shared<opset8::Reshape>(ss_node, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        auto convert = make_shared<opset8::Convert>(squeeze_node, element::Type_t::u32);
        auto reshape_node = make_shared<opset8::Reshape>(scores, opset8::Constant::create(element::i32, Shape{1}, {-1}), true);
        auto gather = make_shared<opset8::Gather>(reshape_node, convert, opset8::Constant::create(element::i32, Shape{1}, {0}));

        f_ref = make_shared<Function>(NodeVector{gather}, ParameterVector{boxes, scores});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, test_convert_to_unsigned_nms_gather_3) {
    // if Gather goes right after NMS no converts should be inserted
    auto boxes = make_shared<opset8::Parameter>(element::f32, Shape{1, 1000, 4});
    auto scores = make_shared<opset8::Parameter>(element::f32, Shape{1, 1, 1000});
    auto nms = make_shared<opset8::NonMaxSuppression>(boxes, scores);

    auto gather = make_shared<opset8::Gather>(nms->output(0), opset8::Constant::create(element::i32, Shape{1}, {2}),
                                              opset8::Constant::create(element::i32, Shape{1}, {1}));

    shared_ptr<Function> f = make_shared<Function>(NodeVector{gather}, ParameterVector{boxes, scores});

    pass::Manager manager;
    manager.register_pass<pass::InitNodeInfo>();
    manager.register_pass<pass::ConvertToUnsignedNmsGather>();
    manager.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));
    ASSERT_EQ(count_ops_of_type<opset1::Convert>(f), 0);
}
