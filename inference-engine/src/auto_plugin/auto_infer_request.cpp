// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "auto_infer_request.hpp"
#include <ie_input_info.hpp>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>

namespace AutoPlugin {
    using namespace InferenceEngine;

AutoInferRequest::AutoInferRequest(const InputsDataMap&   networkInputs,
                                   const OutputsDataMap&  networkOutputs,
                                   const InferRequest&    inferRequest)
    : IInferRequestInternal(networkInputs, networkOutputs)
    , _inferRequest(inferRequest) {
    if (_inferRequest) {
        for (const auto &it : _networkInputs)
            _inputs[it.first] = _inferRequest.GetBlob(it.first);
        for (const auto &it : _networkOutputs)
            _outputs[it.first] = _inferRequest.GetBlob(it.first);
        return;
    }
    IE_THROW(NotAllocated);
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> AutoInferRequest::GetPerformanceCounts() const {
    return _inferRequest.GetPerformanceCounts();
}

void AutoInferRequest::InferImpl() {
    _inferRequest.Infer();
}

void AutoInferRequest::StartAsyncImpl() {
    _inferRequest.StartAsync();
}

StatusCode AutoInferRequest::Wait(int64_t millis_timeout) {
    return _inferRequest.Wait(millis_timeout);
}

void AutoInferRequest::SetCallback(Callback callback) {
    _inferRequest.SetCompletionCallback([callback](){
        callback(nullptr);
    });
}

void AutoInferRequest::SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& data) {
    _inferRequest.SetBlob(name, data);
}

Blob::Ptr AutoInferRequest::GetBlob(const std::string& name) {
    return _inferRequest.GetBlob(name);
}

}  // namespace AutoPlugin
