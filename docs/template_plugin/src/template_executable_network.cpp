// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_metric_helpers.hpp>
#include <ie_plugin_config.hpp>
#include <threading/ie_executor_manager.hpp>

#include "template/template_config.hpp"
#include "template_plugin.hpp"
#include "template_executable_network.hpp"

using namespace TemplatePlugin;

// ! [executable_network:ctor_cnnnetwork]
TemplatePlugin::ExecutableNetwork::ExecutableNetwork(const std::shared_ptr<ngraph::Function>&   function,
                                                     const Configuration&                       cfg,
                                                     const Plugin::Ptr&                         plugin) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault(nullptr, nullptr), // Disable default threads creation
    _cfg(cfg),
    _plugin(plugin),
    _function(function) {
    // TODO: if your plugin supports device ID (more that single instance of device can be on host machine)
    // you should select proper device based on KEY_DEVICE_ID or automatic behavior
    // In this case, _waitExecutor should also be created per device.
    try {
        MapGraph();
        InitExecutor(); // creates thread-based executor using for async requests
    } catch (const InferenceEngineException&) {
        throw;
    } catch (const std::exception & e) {
        THROW_IE_EXCEPTION << "Standard exception from compilation library: " << e.what();
    } catch (...) {
        THROW_IE_EXCEPTION << "Generic exception is thrown";
    }
}
// ! [executable_network:ctor_cnnnetwork]

// ! [executable_network:ctor_import_stream]
TemplatePlugin::ExecutableNetwork::ExecutableNetwork(std::istream &                 model,
                                                     const Configuration&           cfg,
                                                     const Plugin::Ptr&             plugin) :
    _cfg(cfg),
    _plugin(plugin) {
    // TODO: since Import network is not a mandatory functionality, this ctor can just be removed
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}
// ! [executable_network:ctor_import_stream]

// ! [executable_network:map_graph]
void TemplatePlugin::ExecutableNetwork::MapGraph() {
    // TODO: perform actual graph mapping to backend graph representation / kernels

    // Generate backend specific blob mappings. For example Inference Engine uses not ngraph::Result nodes friendly name
    // as inference request output names but the name of the layer before.
    for (auto&& result : _function->get_results()) {
        auto previousOutput = result->get_input_source_output(0);
        auto outputName = previousOutput.get_node()->get_friendly_name();
        if (previousOutput.get_node()->get_output_size() > 1) {
            outputName += '.' + std::to_string(previousOutput.get_index());
        }
        _outputIndex.emplace(outputName, _function->get_result_index(result));
    }
    for (auto&& parameter : _function->get_parameters()) {
        _inputIndex.emplace(parameter->get_friendly_name(), _function->get_parameter_index(parameter));
    }

    // Perform any other steps like allocation and filling device buffers for weights, and so on
}
// ! [executable_network:map_graph]


// ! [executable_network:init_executor]
void TemplatePlugin::ExecutableNetwork::InitExecutor() {
    // Default multi-threaded configuration is balanced for throughtput and latency cases and takes into account
    // real hardware cores and NUMA nodes.
    auto streamsExecutorConfig = InferenceEngine::IStreamsExecutor::Config::MakeDefaultMultiThreaded(_cfg._streamsExecutorConfig);
    streamsExecutorConfig._name = "TemplateStreamsExecutor";
    // As Inference Engine CPU Streams Executor creates some additional therads
    // it is better to avoid threads recreateion as some OSs memory allocator can not manage such usage cases
    // and memory consumption can be larger than it is expected.
    // So Inference Engone provides executors cache.
    _taskExecutor = ExecutorManager::getInstance()->getIdleCPUStreamsExecutor(streamsExecutorConfig);
    // NOTE: callback Executor is not configured. So callback will be called in the thread of tha last stage of inference request pipeline
    // _callbackExecutor = ExecutorManager::getInstance()->getIdleCPUStreamsExecutor({"TemplateCallbackExecutor"});
}
// ! [executable_network:init_executor]


// ! [executable_network:create_infer_request_impl]
InferenceEngine::InferRequestInternal::Ptr TemplatePlugin::ExecutableNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                                                     InferenceEngine::OutputsDataMap networkOutputs) {
    return std::make_shared<TemplateInferRequest>(networkInputs, networkOutputs, std::static_pointer_cast<ExecutableNetwork>(shared_from_this()));
}
// ! [executable_network:create_infer_request_impl]

// ! [executable_network:create_infer_request]
void TemplatePlugin::ExecutableNetwork::CreateInferRequest(IInferRequest::Ptr& asyncRequest) {
    auto internalRequest = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    auto asyncThreadSafeImpl = std::make_shared<TemplateAsyncInferRequest>(std::static_pointer_cast<TemplateInferRequest>(internalRequest),
                                                                           _taskExecutor, _plugin->_waitExecutor, _callbackExecutor);
    asyncRequest.reset(new InferenceEngine::InferRequestBase<TemplateAsyncInferRequest>(asyncThreadSafeImpl),
                       [](InferenceEngine::IInferRequest *p) { p->Release(); });
    asyncThreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
}
// ! [executable_network:create_infer_request]

// ! [executable_network:get_config]
void TemplatePlugin::ExecutableNetwork::GetConfig(const std::string &name, Parameter &result, ResponseDesc *resp) const {
    result = _cfg.Get(name);
}
// ! [executable_network:get_config]

// ! [executable_network:get_metric]
void TemplatePlugin::ExecutableNetwork::GetMetric(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *) const {
    // TODO: return more supported values for metrics
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        result = IE_SET_METRIC(SUPPORTED_METRICS, std::vector<std::string>{
            METRIC_KEY(NETWORK_NAME),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS),
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)});
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        std::vector<std::string> configKeys = {
            CONFIG_KEY(DEVICE_ID),
            CONFIG_KEY(PERF_COUNT),
            TEMPLATE_CONFIG_KEY(THROUGHPUT_STREAMS) };
        auto streamExecutorConfigKeys = IStreamsExecutor::Config{}.SupportedKeys();
        for (auto&& configKey : streamExecutorConfigKeys) {
            configKeys.emplace_back(configKey);
        }
        result = IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (METRIC_KEY(NETWORK_NAME) == name) {
        auto networkName = _function->get_friendly_name();
        result = IE_SET_METRIC(NETWORK_NAME, networkName);
    } else if (METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS) == name) {
        unsigned int value = _cfg._streamsExecutorConfig._streams;
        result = IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, value);
    } else {
        THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork metric: " << name;
    }
}
// ! [executable_network:get_metric]

// ! [executable_network:export_impl]
void TemplatePlugin::ExecutableNetwork::ExportImpl(std::ostream& dlaModel) {
    // TODO: Code which exports graph from std::ostream
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}
// ! [executable_network:export_impl]
