// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>

#include <ie_parallel.hpp>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <threading/ie_itask_executor.hpp>

#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
# include <tbb/concurrent_queue.h>
#endif

namespace AutoPlugin {

class AutoInferencePlugin;

using DeviceName = std::string;
using ConfigType = std::map<std::string, std::string>;
using NetworkFuture = std::future<InferenceEngine::SoExecutableNetworkInternal>;

template<typename T>
using DeviceMap = std::unordered_map<DeviceName, T>;

#if ((IE_THREAD == IE_THREAD_TBB) || (IE_THREAD == IE_THREAD_TBB_AUTO))
template <typename T>
using ThreadSafeQueue = tbb::concurrent_queue<T>;
template <typename T>
using ThreadSafeBoundedQueue = tbb::concurrent_bounded_queue<T>;
#else
template <typename T>
class ThreadSafeQueue {
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(_mutex);
        _queue.push(std::move(value));
    }
    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_queue.empty()) {
            value = std::move(_queue.front());
            _queue.pop();
            return true;
        } else {
            return false;
        }
    }
private:
    std::queue<T>   _queue;
    std::mutex      _mutex;
};
template <typename T>
class ThreadSafeBoundedQueue {
public:
    ThreadSafeBoundedQueue() = default;
    bool try_push(T value) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_capacity > _queue.size()) {
            _queue.push(std::move(value));
            return true;
        }
        return false;
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_capacity && !_queue.empty()) {
            value = std::move(_queue.front());
            _queue.pop();
            return true;
        } else {
            return false;
        }
    }
    void set_capacity(std::size_t newCapacity) {
        std::lock_guard<std::mutex> lock(_mutex);
        _capacity = newCapacity;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.size();
    }

private:
    std::queue<T>   _queue;
    mutable std::mutex      _mutex;
    std::size_t     _capacity { 0 };
};
#endif

class AutoExecutableNetwork : public InferenceEngine::IExecutableNetworkInternal,
                              public InferenceEngine::ITaskExecutor {
public:
    using Ptr = std::shared_ptr<AutoExecutableNetwork>;
    struct WorkerInferRequest {
        InferenceEngine::SoIInferRequestInternal  _inferRequest;
        InferenceEngine::Task                     _task;
        std::exception_ptr                        _exceptionPtr = nullptr;
    };
    using NotBusyWorkerRequests = ThreadSafeBoundedQueue<WorkerInferRequest*>;

    AutoExecutableNetwork(const std::string&                 modelPath,
                          const InferenceEngine::CNNNetwork& network,
                          const ConfigType&                  config,
                          AutoInferencePlugin*               plugin);

    void Export(std::ostream& networkModel) override;
    InferenceEngine::RemoteContext::Ptr GetContext() const override;
    InferenceEngine::CNNNetwork GetExecGraphInfo() override;
    InferenceEngine::Parameter GetMetric(const std::string &name) const override;
    void SetConfig(const std::map<std::string, InferenceEngine::Parameter>& config) override;
    InferenceEngine::Parameter GetConfig(const std::string& name) const override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                       InferenceEngine::OutputsDataMap networkOutputs) override;
    void run(InferenceEngine::Task inferTask) override;
    bool TryGetActualNetwork(InferenceEngine::SoExecutableNetworkInternal& soExecNetwork);

    ~AutoExecutableNetwork() final;

    // todo: if pipeline task could have parameters, this could be not needed.
    static thread_local WorkerInferRequest*                     _thisWorkerInferRequest;

private:
    void WaitForActualDevice() const;
    void ScheduleToWorkerInferRequest(InferenceEngine::Task, DeviceName preferred_device = "");

private:
    InferenceEngine::SoExecutableNetworkInternal _networkFirstReady;
    mutable InferenceEngine::SoExecutableNetworkInternal _networkActualNeeded;
    NetworkFuture _cpuFuture;
    mutable NetworkFuture _acceleratorFuture;
    bool _enablePerfCount;
    mutable std::atomic<bool> _alreadyActualNetwork = {false};
    std::map<std::string, InferenceEngine::Parameter> _cacheConfig;
    AutoInferencePlugin* _autoPlugin;
    DeviceMap<std::vector<WorkerInferRequest>> _workerRequests;
    DeviceMap<NotBusyWorkerRequests>           _idleWorkerRequests;
    ThreadSafeQueue<InferenceEngine::Task>                      _inferPipelineTasks;
    std::string _cpuDeviceName;
    std::string _acceleratorDeviceName;
};

}  // namespace AutoPlugin
