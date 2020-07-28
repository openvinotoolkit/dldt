// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include <ie_icnn_network.hpp>

class MockPlugin : public InferenceEngine::InferencePluginInternal {
    InferenceEngine::IInferencePluginInternal * _target = nullptr;

public:
    explicit MockPlugin(InferenceEngine::IInferencePluginInternal*target);

    void LoadNetwork(InferenceEngine::IExecutableNetwork::Ptr &ret, const InferenceEngine::ICNNNetwork &network,
                     const std::map<std::string, std::string> &config) override;
    ExecutableNetworkInternal::Ptr
    LoadExeNetworkImpl(const InferenceEngine::ICNNNetwork& network,
                       const std::map<std::string, std::string>& config);
};
