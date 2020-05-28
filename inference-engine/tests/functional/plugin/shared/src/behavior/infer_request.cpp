// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>
#include <cpp_interfaces/exception2status.hpp>
#include <thread>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "behavior/infer_request.hpp"

namespace LayerTestsDefinitions {
    std::string InferRequestTests::getTestCaseName(testing::TestParamInfo<InferRequestParams> obj) {
        InferenceEngine::Precision  netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(netPrecision, targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        if (!configuration.empty()) {
            result << "configItem=" << configuration.begin()->first << "_" << configuration.begin()->second;
        }
        return result.str();
    }

    void InferRequestTests::SetUp() {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void InferRequestTests::TearDown() {
        if (targetDevice.find(CommonTestUtils::DEVICE_GPU) != std::string::npos) {
            PluginCache::get().reset();
        }
    }

// Setting empty config to LoadNetwork doesn't throw
TEST_P(InferRequestTests, SetEmptyConfig) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    InferenceEngine::IExecutableNetwork::Ptr execNet;
    std::map<std::string, std::string> config {};
    if (targetDevice.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
    targetDevice.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
        ASSERT_NO_THROW(ie->SetConfig(configuration, targetDevice));
        ASSERT_NO_THROW(execNet = ie->LoadNetwork(cnnNet, targetDevice, config));

    } else {
        ASSERT_THROW(ie->SetConfig(configuration, targetDevice),
                InferenceEngine::details::InferenceEngineException);
        ASSERT_NO_THROW(execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration));
    }
    function.reset();
}

// Load correct network to Plugin to get executable network
TEST_P(InferRequestTests, canLoadCorrectNetworkToGetExecutable) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    InferenceEngine::IExecutableNetwork::Ptr execNet;
    ASSERT_NO_THROW(execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration));
    function.reset();
}

TEST_P(InferRequestTests,  CanCreateTwoExeNetworks) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    InferenceEngine::IExecutableNetwork::Ptr execNet;
    for (auto i = 0; i < 2; i++) {
        ASSERT_NO_THROW(execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration));
    }
    function.reset();
}

TEST_P(InferRequestTests, CanCreateInferRequest) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    function.reset();
}

TEST_P(InferRequestTests, failToSetNullptrForInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr inputBlob = nullptr;
    ASSERT_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob),
                     InferenceEngine::details::InferenceEngineException);
    function.reset();
}

TEST_P(InferRequestTests, failToSetEmptyInputBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob;
    ASSERT_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob),
                     InferenceEngine::details::InferenceEngineException);
    function.reset();
}

TEST_P(InferRequestTests, failToSetEmptyOutputBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob;
    ASSERT_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob),
            InferenceEngine::details::InferenceEngineException);
    function.reset();
}

TEST_P(InferRequestTests, failToSetNotAllocatedInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
    function.reset();
}

TEST_P(InferRequestTests, failToSetNotAllocatedOutput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob));
    function.reset();
}

TEST_P(InferRequestTests, failToSetBlobWithIncorrectName) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    const char incorrect_input_name[] = "incorrect_input_name";
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    blob->allocate();
    ASSERT_THROW(req.SetBlob(incorrect_input_name, blob),
                    InferenceEngine::details::InferenceEngineException);
    function.reset();
}

TEST_P(InferRequestTests, failToSetInputWithIncorrectSizes) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    blob->allocate();
    blob->getTensorDesc().getDims()[0]*=2;
    ASSERT_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob),
            InferenceEngine::details::InferenceEngineException);
    function.reset();
}

TEST_P(InferRequestTests, failToSetOutputWithIncorrectSizes) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    blob->allocate();
    blob->getTensorDesc().getDims()[0]*=2;
    ASSERT_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob),
            InferenceEngine::details::InferenceEngineException);
    function.reset();
}

TEST_P(InferRequestTests, canInferWithoutSetAndGetInOut) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.Infer());
    function.reset();
}

TEST_P(InferRequestTests, canProcessDeallocatedInputBlobAfterGetBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(req.Infer());
    function.reset();
}

TEST_P(InferRequestTests, canProcessDeallocatedInputBlobAfterGetBlobForAsync) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW(req.StartAsync());
    function.reset();
}

TEST_P(InferRequestTests, canProcessDeallocatedInputBlobAfterGetAndSetBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW(req.StartAsync());
    function.reset();
}

TEST_P(InferRequestTests, canProcessDeallocatedInputBlobAfterSetBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW(req.Infer(), InferenceEngine::details::InferenceEngineException);
    function.reset();
}

TEST_P(InferRequestTests, canProcessDeallocatedOutputBlobAfterGetBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW(req.Infer(), InferenceEngine::details::InferenceEngineException);
    function.reset();
}

TEST_P(InferRequestTests, canProcessDeallocatedOutputBlobAfterGetBlobForAsync) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW(req.Infer(), InferenceEngine::details::InferenceEngineException);
    ASSERT_THROW(req.StartAsync(), InferenceEngine::details::InferenceEngineException);
    function.reset();
}

TEST_P(InferRequestTests, canProcessDeallocatedOutputBlobAfterGetAndSetBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW(req.Infer(), InferenceEngine::details::InferenceEngineException);
    function.reset();
}

TEST_P(InferRequestTests, canProcessDeallocatedOutputBlobAfterSetBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW(req.Infer(), InferenceEngine::details::InferenceEngineException);
    function.reset();
}

TEST_P(InferRequestTests, secondCallGetOutputDoNotReAllocateData) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob1;
    InferenceEngine::Blob::Ptr blob2;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob1 = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(blob2 = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_EQ(blob1.get(), blob2.get());
    function.reset();
}

TEST_P(InferRequestTests, CorrectOneAsyncInferWithGetInOutWithInfWait) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    req.Infer();
    req.StartAsync();
    InferenceEngine::StatusCode sts;
    sts = req.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    function.reset();
}

// Plugin correct infer request with allocating input and result BlobMaps inside plugin
TEST_P(InferRequestTests, canStartAsyncInferWithGetInOutWithStatusOnlyWait) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    req.Infer();
    req.StartAsync();
    InferenceEngine::StatusCode sts;
    sts = req.Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
    ASSERT_TRUE(sts == InferenceEngine::StatusCode::OK ||
        InferenceEngine::StatusCode::RESULT_NOT_READY);
    function.reset();
}

// Plugin correct infer request with allocating input and result BlobMaps inside plugin
TEST_P(InferRequestTests, FailedAsyncInferWithNegativeTimeForWait) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    req.Infer();
    req.StartAsync();
    ASSERT_THROW(req.Wait(-2), InferenceEngine::details::InferenceEngineException);
    function.reset();
}

TEST_P(InferRequestTests, canRun3SyncRequestsConsistentlyFromThreads) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    auto req1 = execNet.CreateInferRequest();
    auto req2 = execNet.CreateInferRequest();
    auto req3 = execNet.CreateInferRequest();
    InferenceEngine::ResponseDesc response1, response2, response3;
    InferenceEngine::StatusCode sts1, sts2, sts3;

    std::thread t1([&] { req1.Infer(); sts1 = req1.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY); });
    std::thread t2([&] { req2.Infer(); sts2 = req2.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY); });
    std::thread t3([&] { req3.Infer(); sts3 = req3.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY); });

    t1.join();
    t2.join();
    t3.join();

    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), sts1) << response1.msg;
    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), sts2) << response2.msg;
    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), sts3) << response3.msg;

    function.reset();
}

TEST_P(InferRequestTests, canRun3AsyncRequestsConsistentlyWithWait) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    auto req1 = execNet.CreateInferRequest();
    auto req2 = execNet.CreateInferRequest();
    auto req3 = execNet.CreateInferRequest();
    InferenceEngine::ResponseDesc response1, response2, response3;
    InferenceEngine::StatusCode sts1, sts2, sts3;

    req1.StartAsync();
    ASSERT_NO_THROW(req1.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY));

    req2.Infer();
    ASSERT_NO_THROW(req2.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY));

    req3.Infer();
    ASSERT_NO_THROW(req3.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY));

    function.reset();
}

TEST_P(InferRequestTests, canRun3AsyncRequestsConsistentlyFromThreadsWithoutWait) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    auto req1 = execNet.CreateInferRequest();
    auto req2 = execNet.CreateInferRequest();
    auto req3 = execNet.CreateInferRequest();
    InferenceEngine::ResponseDesc response1, response2, response3;
    InferenceEngine::StatusCode sts1, sts2, sts3;

    req1.Infer();
    req2.Infer();
    req3.Infer();

    std::thread t1([&] { req1.StartAsync(); sts1 = req1.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY); });
    std::thread t2([&] { req2.StartAsync(); sts2 = req2.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY); });
    std::thread t3([&] { req3.StartAsync(); sts3 = req3.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY); });

    t1.join();
    t2.join();
    t3.join();

    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), sts1) << response1.msg;
    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), sts2) << response2.msg;
    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), sts3) << response3.msg;

    function.reset();
}

TEST_P(InferRequestTests, canWaitWithotStartAsync) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    ASSERT_NO_THROW(req.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY));
    ASSERT_NO_THROW(req.Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY));
    ASSERT_NO_THROW(req.Wait(1));
    function.reset();
}

TEST_P(InferRequestTests, returnDeviceBusyOnSetBlobAfterAsyncInfer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto&& config = configuration;
    auto itConfig = config.find(CONFIG_KEY(CPU_THROUGHPUT_STREAMS));
    if (itConfig != config.end()) {
        if (itConfig->second != "CPU_THROUGHPUT_AUTO") {
            if (std::stoi(itConfig->second) == 0) {
                GTEST_SKIP() << "Not applicable with disabled streams";
            }
        }
    }
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    auto outputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    InferenceEngine::ResponseDesc response;

    InferenceEngine::StatusCode sts;
    sts = req.Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
    ASSERT_EQ(InferenceEngine::StatusCode::INFER_NOT_STARTED, sts) << response.msg;
    req.StartAsync();
    sts = req.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), sts) << response.msg;
    try {
        req.SetBlob(cnnNet.getInputsInfo().begin()->first, outputBlob);
    }
    catch (const std::exception &e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
    sts = req.Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
    ASSERT_TRUE(sts == InferenceEngine::StatusCode::OK ||
        sts == InferenceEngine::StatusCode::RESULT_NOT_READY) << response.msg;

    function.reset();
}

TEST_P(InferRequestTests, returnDeviceBusyOnGetBlobAfterAsyncInfer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    auto outputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    InferenceEngine::ResponseDesc response;
    InferenceEngine::StatusCode sts;
    req.StartAsync();
    sts = req.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), sts) << response.msg;
    try {
        req.SetBlob(cnnNet.getInputsInfo().begin()->first, outputBlob);
    }
    catch (const std::exception &e) {
        std::cout << "Exception" << e.what() << std::endl;
    }
    function.reset();
}

TEST_P(InferRequestTests, returnDeviceBusyOnGetPerformanceCountAfterAsyncInfer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    auto outputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    InferenceEngine::ResponseDesc response;
    InferenceEngine::StatusCode sts;
    req.StartAsync();
    sts = req.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), sts) << response.msg;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;

    try {
        perfMap = req.GetPerformanceCounts();
    }
    catch (const std::exception &e) {
        std::cout << "Exception" << e.what() << std::endl;
    }
    function.reset();
}
}  // namespace LayerTestsDefinitions
