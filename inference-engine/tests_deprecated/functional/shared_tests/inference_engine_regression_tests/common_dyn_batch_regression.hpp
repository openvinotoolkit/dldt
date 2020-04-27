// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <regression_tests.hpp>
#include <string>
#include <ngraph_functions/subgraph_builders.hpp>
#include <functional_test_utils/blob_utils.hpp>

using namespace ::testing;
using namespace InferenceEngine;

class CommonDynBatchFuncTestParams {
public:
    std::string deviceName;
    double nearValue;
    int batch_limit;
    int cur_batch;

    CommonDynBatchFuncTestParams(const std::string& _deviceName,
                                 int blimit,
                                 int batch,
                                 double _nearValue = 0.01f):
            deviceName(_deviceName),
			batch_limit(blimit),
			cur_batch(batch),
            nearValue(_nearValue)
    {}
};

template <Precision::ePrecision P>
class TestNoRegressionDynBatch : public Regression::RegressionTests, public WithParamInterface<CommonDynBatchFuncTestParams> {
    std::string getDeviceName() const override {
        return GetParam().deviceName;
    }

public:
    double getNearValue() {
        return GetParam().nearValue;
    }
    int get_batch_limit() {
        return GetParam().batch_limit;
    }
    int get_cur_batch() {
        return GetParam().cur_batch;
    }
};

using TestNoRegressionDynBatchFP32 = TestNoRegressionDynBatch<Precision::FP32>;

TEST_P(TestNoRegressionDynBatchFP32, dynBatch) {
    int bl = get_batch_limit();
    int bsz = get_cur_batch();
    auto fnPtr = ngraph::builder::subgraph::makeSingleConv({static_cast<size_t>(bl), 4, 20, 20});

    CNNNetwork net(fnPtr);
    auto ieCore = PluginCache::get().ie();
    InferenceEngine::ExecutableNetwork exeNet = ieCore->LoadNetwork(net, GetParam().deviceName, {{PluginConfigParams::KEY_DYN_BATCH_ENABLED,
                                                                                           PluginConfigParams::YES}});
    InferenceEngine::InferRequest inferRequest = exeNet.CreateInferRequest();

    auto blob = FuncTestUtils::createAndFillBlob(net.getInputsInfo().begin()->second->getTensorDesc());

    inferRequest.SetBatch(bsz);
    inferRequest.SetBlob(net.getInputsInfo().begin()->first, blob);
    inferRequest.Infer();
    auto *outRawData = inferRequest.GetBlob(net.getOutputsInfo().begin()->first)->cbuffer().as<float *>();

    auto refOutData = ngraph::helpers::inferFnWithInterp<ngraph::element::Type_t::f32>(fnPtr,
                                                                                       {blob->cbuffer().as<float *>()});

    auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
    std::vector<size_t> inShapeLimited{size_t(bsz), 4, 20, 20};
    size_t outElementsCount = std::accumulate(begin(inShapeLimited), end(inShapeLimited), 1, std::multiplies<size_t>());
    FuncTestUtils::compareRawBuffers(outRawData, *refOutData[0], outElementsCount, outElementsCount, thr);
    if (GetParam().deviceName.find(CommonTestUtils::DEVICE_GPU) != std::string::npos) {
        PluginCache::get().reset();
    }
}

std::string  getTestCaseName(TestParamInfo<CommonDynBatchFuncTestParams> obj) {
    return obj.param.deviceName + "_" + std::to_string(obj.param.batch_limit)
        + "_" + std::to_string(obj.param.cur_batch);
}
