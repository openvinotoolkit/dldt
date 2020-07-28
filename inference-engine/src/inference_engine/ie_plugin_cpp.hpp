// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the Inference Engine plugin C++ API
 *
 * @file ie_plugin_cpp.hpp
 */
#pragma once

#include <map>
#include <memory>
#include <string>

#include "file_utils.h"
#include "cpp/ie_executable_network.hpp"
#include "cpp/ie_cnn_network.h"
#include "details/ie_exception_conversion.hpp"
#include "ie_plugin_ptr.hpp"

#define CALL_RETURN_FNC(function, ...)         \
    if (!actual) THROW_IE_EXCEPTION << "Wrapper used in the CALL_RETURN_FNC was not initialized."; \
    return actual->function(__VA_ARGS__);

#define CALL_FNC(function, ...)         \
    if (!actual) THROW_IE_EXCEPTION << "Wrapper used in the CALL_RETURN_FNC was not initialized."; \
    actual->function(__VA_ARGS__);

namespace InferenceEngine {

/**
 * @brief This class is a C++ API wrapper for IInferencePlugin.
 *
 * It can throw exceptions safely for the application, where it is properly handled.
 */
class InferencePlugin {
    InferenceEnginePluginPtr actual;

public:
    /** @brief A default constructor */
    InferencePlugin() = default;

    /**
     * @brief Constructs a plugin instance from the given pointer.
     *
     * @param pointer Initialized Plugin pointer
     */
    explicit InferencePlugin(const InferenceEnginePluginPtr& pointer): actual(pointer) {
        if (actual == nullptr) {
            THROW_IE_EXCEPTION << "InferencePlugin wrapper was not initialized.";
        }
    }

    explicit InferencePlugin(const FileUtils::FilePath & libraryLocation) :
        actual(libraryLocation) {
        if (actual == nullptr) {
            THROW_IE_EXCEPTION << "InferencePlugin wrapper was not initialized.";
        }
    }

    void SetName(const std::string & deviceName) {
        CALL_FNC(SetName, deviceName);
    }

    void SetCore(ICore* core) {
        CALL_FNC(SetCore, core);
    }

    /**
     * @copybrief IInferencePlugin::GetVersion
     *
     * Wraps IInferencePlugin::GetVersion
     * @return A plugin version
     */
    const Version GetVersion() const noexcept {
        CALL_RETURN_FNC(GetVersion);
    }

    /**
     * @copybrief InferencePlugin::LoadNetwork
     *
     * Wraps IInferencePlugin::LoadNetwork
     * @param network A network object to load
     * @param config A map of configuration options
     * @return Created Executable Network object
     */
    ExecutableNetwork LoadNetwork(CNNNetwork network, const std::map<std::string, std::string>& config) {
        IExecutableNetwork::Ptr ret;
        CALL_FNC(LoadNetwork, ret, network, config);
        if (ret.get() == nullptr) THROW_IE_EXCEPTION << "Internal error: pointer to executable network is null";
        return ExecutableNetwork(ret, actual);
    }

    /**
     * @copybrief IInferencePlugin::AddExtension
     *
     * Wraps IInferencePlugin::AddExtension
     *
     * @param extension Pointer to loaded Extension
     */
    void AddExtension(InferenceEngine::IExtensionPtr extension) {
        CALL_FNC(AddExtension, extension);
    }

    /**
     * @copybrief IInferencePlugin::SetConfig
     *
     * Wraps IInferencePlugin::SetConfig
     * @param config A configuration map
     */
    void SetConfig(const std::map<std::string, std::string>& config) {
        CALL_FNC(SetConfig, config);
    }

    /**
     * @copybrief IInferencePlugin::ImportNetwork
     *
     * Wraps IInferencePlugin::ImportNetwork
     * @param modelFileName A path to the imported network
     * @param config A configuration map
     * @return Created Executable Network object
     */
    ExecutableNetwork ImportNetwork(const std::string& modelFileName,
                                    const std::map<std::string, std::string>& config) {
        if (!actual) THROW_IE_EXCEPTION << "Wrapper used in the CALL_RETURN_FNC was not initialized."; \
        auto ret = actual->ImportNetwork(modelFileName, config);
        return ExecutableNetwork(ret, actual);
    }

    /**
     * @copybrief IInferencePlugin::QueryNetwork
     *
     * Wraps IInferencePlugin::QueryNetwork
     *
     * @param network A network object to query
     * @param config A configuration map
     * @param res Query results
     */
    void QueryNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config,
                      QueryNetworkResult& res) const {
        if (actual == nullptr) THROW_IE_EXCEPTION << "InferencePlugin wrapper was not initialized";
        actual->QueryNetwork(network, config, res);
        if (res.rc != OK) THROW_IE_EXCEPTION << res.resp.msg;
    }

    ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                    const std::map<std::string, std::string> &config) {
        CALL_RETURN_FNC(ImportNetwork, networkModel, config);
    }

    Parameter GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const {
        CALL_RETURN_FNC(GetMetric, name, options);
    }

    ExecutableNetwork LoadNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config,
                                  RemoteContext::Ptr context) {
        CALL_RETURN_FNC(LoadNetwork, network, config, context);
    }

    RemoteContext::Ptr CreateContext(const ParamMap& params) {
        CALL_RETURN_FNC(CreateContext, params);
    }

    RemoteContext::Ptr GetDefaultContext() {
        CALL_RETURN_FNC(GetDefaultContext);
    }

    ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                    const RemoteContext::Ptr& context,
                                    const std::map<std::string, std::string>& config) {
        CALL_RETURN_FNC(ImportNetwork, networkModel, context, config);
    }

    Parameter GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const {
        CALL_RETURN_FNC(GetConfig, name, options);
    }

    /**
     * @brief Converts InferenceEngine to InferenceEnginePluginPtr pointer
     *
     * @return Wrapped object
     */
    operator InferenceEngine::InferenceEnginePluginPtr() {
        return actual;
    }

    /**
     * @brief Shared pointer on InferencePlugin object
     *
     */
    using Ptr = std::shared_ptr<InferencePlugin>;
};
}  // namespace InferenceEngine
