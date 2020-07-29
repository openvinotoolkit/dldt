// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_utils.h"
#include "../common/tests_utils.h"
#include "../common/managers/thread_manager.h"
#include "tests_pipelines/tests_pipelines.h"

#include <gtest/gtest.h>
#include <chrono>

#include <inference_engine.hpp>

using namespace InferenceEngine;


class MemCheckTestSuite : public ::testing::TestWithParam<TestCase> {
public:
    std::string test_name, model, model_name, device;
    TestReferences test_refs;

    void SetUp() override {
        const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        test_name = std::string(test_info->name()).substr(0, std::string(test_info->name()).find('/'));
        //const std::string full_test_name = std::string(test_info->test_case_name()) + "." + std::string(test_info->name());

        const auto& test_params = GetParam();
        model = test_params.model;
        model_name = test_params.model_name;
        device = test_params.device;

        test_refs.collect_vm_values_for_test(test_name, test_params);
        if (!Environment::Instance().getCollectResultsOnly()) {
            ASSERT_GT(test_refs.references[VMSIZE], 0) << "Reference value of VmSize is less than 0. Value: "
                                               << test_refs.references[VMSIZE];
            ASSERT_GT(test_refs.references[VMPEAK], 0) << "Reference value of VmPeak is less than 0. Value: "
                                               << test_refs.references[VMPEAK];
            ASSERT_GT(test_refs.references[VMRSS], 0) << "Reference value of VmRSS is less than 0. Value: "
                                              << test_refs.references[VMRSS];
            ASSERT_GT(test_refs.references[VMHWM], 0) << "Reference value of VmHWM is less than 0. Value: "
                                              << test_refs.references[VMHWM];
        }
    }
};

// tests_pipelines/tests_pipelines.cpp
TEST_P(MemCheckTestSuite, create_exenetwork) {
    log_info("Create ExecutableNetwork from network: \"" << model
                                                         << "\" for device: \"" << device << "\"");
    auto test_pipeline = [&]{
        MemCheckPipeline memCheckPipeline;
        std::chrono::high_resolution_clock::time_point t1, t2;

        Core ie;

        t1 = std::chrono::high_resolution_clock::now();
        ie.GetVersions(device);
        t2 = std::chrono::high_resolution_clock::now();
        auto tm_duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        log_info("Time of load plugin: " << tm_duration);

        t1 = std::chrono::high_resolution_clock::now();
        CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        t2 = std::chrono::high_resolution_clock::now();
        tm_duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        log_info("Time of read network: " << tm_duration);

        t1 = std::chrono::high_resolution_clock::now();
        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, device);
        t2 = std::chrono::high_resolution_clock::now();
        tm_duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        log_info("Time of load_network: " << tm_duration);

        log_info("Memory consumption after LoadNetwork:");
        memCheckPipeline.record_measures(test_name);

        log_debug(memCheckPipeline.get_reference_record_for_test(test_name, model_name, device));
        return memCheckPipeline.measure();
    };

    TestResult res = common_test_pipeline(test_pipeline, test_refs.references);
    EXPECT_EQ(res.first, TestStatus::TEST_OK) << res.second;
}

TEST_P(MemCheckTestSuite, infer_request_inference) {
    log_info("Inference of InferRequest from network: \"" << model
                                                          << "\" for device: \"" << device << "\"");
    auto test_pipeline = [&]{
        MemCheckPipeline memCheckPipeline;
        std::chrono::high_resolution_clock::time_point t1, t2;

        Core ie;

        t1 = std::chrono::high_resolution_clock::now();
        ie.GetVersions(device);
        t2 = std::chrono::high_resolution_clock::now();
        auto tm_duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        log_info("Time of load plugin: " << tm_duration);

        t1 = std::chrono::high_resolution_clock::now();
        CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        t2 = std::chrono::high_resolution_clock::now();
        tm_duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        log_info("Time of read network: " << tm_duration);

        t1 = std::chrono::high_resolution_clock::now();
        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, device);
        t2 = std::chrono::high_resolution_clock::now();
        tm_duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        log_info("Time of load_network: " << tm_duration);

        t1 = std::chrono::high_resolution_clock::now();
        InferRequest inferRequest = exeNetwork.CreateInferRequest();
        inferRequest.Infer();
        OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
        for (auto &output : output_info)
            Blob::Ptr outputBlob = inferRequest.GetBlob(output.first);
        t2 = std::chrono::high_resolution_clock::now();
        tm_duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        log_info("Time of inference: " << tm_duration);

        log_info("Memory consumption after Inference:");
        memCheckPipeline.record_measures(test_name);

        log_debug(memCheckPipeline.get_reference_record_for_test(test_name, model_name, device));
        return memCheckPipeline.measure();
    };

    TestResult res = common_test_pipeline(test_pipeline, test_refs.references);
    EXPECT_EQ(res.first, TestStatus::TEST_OK) << res.second;
}
// tests_pipelines/tests_pipelines.cpp

INSTANTIATE_TEST_CASE_P(MemCheckTests, MemCheckTestSuite,
                        ::testing::ValuesIn(
                                generateTestsParams({"devices", "models"})),
                        getTestCaseName);
