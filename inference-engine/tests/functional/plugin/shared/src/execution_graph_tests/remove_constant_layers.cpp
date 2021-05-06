// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <execution_graph_tests/remove_constant_layers.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <inference_engine.hpp>

namespace ExecutionGraphTests {

std::string ExecGraphRemoveConstantLayers::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::ostringstream result;
    std::string targetDevice = obj.param;
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

TEST_P(ExecGraphRemoveConstantLayers, RemoveConstantLayers) {
    auto deviceName = this->GetParam();
    ngraph::Shape shape = {3, 2};
    ngraph::element::Type type = ngraph::element::f32;

    using std::make_shared;
    using namespace ngraph::opset5;

    //        in1   in2 //
    //        | \  /    //
    //        |  mul    //
    //        | /       //
    // const  sum       //
    //      \ |         //
    //      sum         //
    //       |          //
    //      out         //

    auto input1 = make_shared<Parameter>(type, shape);
    auto input2 = make_shared<Parameter>(type, shape);
    auto mul   = make_shared<ngraph::op::v1::Multiply>(input1, input2);
    auto sum1  = make_shared<ngraph::op::v1::Add>(mul, input1);
    auto con   = make_shared<Constant>(type, shape, 1);
    auto sum2  = make_shared<ngraph::op::v1::Add>(con, sum1);

    auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector {sum2},
            ngraph::ParameterVector{input1, input2},
            "SimpleNet");

    auto ie  = InferenceEngine::Core();
    auto net = InferenceEngine::CNNNetwork(function);
    auto exec_net   = ie.LoadNetwork(net, deviceName);
    auto exec_graph = exec_net.GetExecGraphInfo();
    auto exec_ops   = exec_graph.getFunction()->get_ops();

    bool checkConst = true;
    for (auto &node : exec_ops) {
        auto var = node->get_rt_info()["layerType"];
        auto s_val = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(var);
        if (s_val->get() == "Const") {
            checkConst = false;
            break;
        }
    }
    ASSERT_TRUE(checkConst);
}

} // namespace ExecutionGraphTests
