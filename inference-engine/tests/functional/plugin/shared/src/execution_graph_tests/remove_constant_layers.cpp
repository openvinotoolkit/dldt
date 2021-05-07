// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <execution_graph_tests/remove_constant_layers.hpp>
#include "functional_test_utils/skip_tests_config.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <inference_engine.hpp>

namespace ExecutionGraphTests {

std::string ExecGraphRemoveConstantLayers::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::ostringstream result;
    std::string targetDevice = obj.param;
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

TEST_P(ExecGraphRemoveConstantLayers, RemoveConstantLayers) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto deviceName = this->GetParam();
    ngraph::Shape shape = {3, 2};
    ngraph::element::Type type = ngraph::element::f32;

    using std::make_shared;
    using namespace ngraph::opset7;

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

    std::map<std::string, std::string> config = {{CONFIG_KEY_INTERNAL(DUMP_CONSTANT_NODES), CONFIG_VALUE(NO)}};

    auto ie  = InferenceEngine::Core();
    auto net = InferenceEngine::CNNNetwork(function);
    auto execNet   = ie.LoadNetwork(net, deviceName, config);
    auto execGraph = execNet.GetExecGraphInfo();
    auto execOps   = execGraph.getFunction()->get_ops();

    bool checkConst = true;
    for (auto &node : execOps) {
        auto var = node->get_rt_info()["layerType"];
        auto val = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(var);
        if (val->get() == "Const") {
            checkConst = false;
            break;
        }
    }
    ASSERT_TRUE(checkConst);
}

} // namespace ExecutionGraphTests
