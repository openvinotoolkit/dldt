// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>
#include <vector>
#include <utility>
#include <memory>
#include <string>

#include "auto_infer_request.hpp"
#include "auto_exec_network.hpp"

namespace AutoPlugin {

class AutoAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<AutoAsyncInferRequest>;

    explicit AutoAsyncInferRequest(const AutoInferRequest::Ptr&                inferRequest,
                                   const AutoExecutableNetwork::Ptr&           autoExecutableNetwork,
                                   const InferenceEngine::ITaskExecutor::Ptr&  callbackExecutor,
                                   bool                                        enablePerfCount);
    void Infer_ThreadUnsafe() override;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;
    ~AutoAsyncInferRequest();

private:
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>  _perfMap;
    AutoInferRequest::Ptr                                               _inferRequest;
    AutoExecutableNetwork::WorkerInferRequest*                          _workerInferRequest = nullptr;
    bool                                                                _enablePerfCount;
};

}  // namespace AutoPlugin
