// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <boost/type_index.hpp>

#include <string>
#include <vector>

#include <cpp/ie_infer_request.hpp>
#include <ie_common.h>

#include "pyopenvino/inference_engine/common.hpp"
#include "pyopenvino/inference_engine/ie_executable_network.hpp"
#include "pyopenvino/inference_engine/ie_infer_request.hpp"
#include "pyopenvino/inference_engine/ie_preprocess_info.hpp"

namespace py = pybind11;

void regclass_InferRequest(py::module m)
{
    py::class_<InferenceEngine::InferRequest, std::shared_ptr<InferenceEngine::InferRequest>> cls(
        m, "InferRequest");

    cls.def("set_batch", &InferenceEngine::InferRequest::SetBatch, py::arg("size"));

    cls.def("get_blob", &InferenceEngine::InferRequest::GetBlob);

    cls.def("set_blob",
            [](InferenceEngine::InferRequest& self, const std::string& name, py::handle blob) {
                self.SetBlob(name, Common::convert_to_blob(blob));
            });

    cls.def("set_input", [](InferenceEngine::InferRequest& self, const py::dict& inputs) {
        for (auto&& input : inputs) {
            auto name = input.first.cast<std::string>();
            auto blob = Common::cast_to_blob(input.second);
            self.SetBlob(name, blob);
        }
    });

    cls.def("set_output", [](InferenceEngine::InferRequest& self, const py::dict& results) {
        for (auto&& result : results) {
            auto name = result.first.cast<std::string>();
            auto blob = Common::cast_to_blob(result.second);
            self.SetBlob(name, blob);
        }
    });

    cls.def("infer", &InferenceEngine::InferRequest::Infer);

    cls.def("async_infer",
            &InferenceEngine::InferRequest::StartAsync,
            py::call_guard<py::gil_scoped_release>());

    cls.def("set_blob", [](InferenceEngine::InferRequest& self,
                           const std::string& name,
                           py::handle blob) {
        self.SetBlob(name,  Common::cast_to_blob(blob));
    });

    cls.def("set_blob", [](InferenceEngine::InferRequest& self,
                           const std::string& name,
                           py::handle blob,
                           const InferenceEngine::PreProcessInfo& info) {
        self.SetBlob(name, Common::cast_to_blob(blob));
    });

    cls.def("wait",
            &InferenceEngine::InferRequest::Wait,
            py::arg("millis_timeout") = InferenceEngine::IInferRequest::WaitMode::RESULT_READY,
            py::call_guard<py::gil_scoped_acquire>());

    cls.def("set_completion_callback",
            [](InferenceEngine::InferRequest* self, py::function f_callback) {
                self->SetCompletionCallback([f_callback]() {
                    py::gil_scoped_acquire acquire;
                    f_callback();
                    py::gil_scoped_release release;
                });
            });

    cls.def("get_perf_counts", [](InferenceEngine::InferRequest& self) {
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
        perfMap = self.GetPerformanceCounts();
        py::dict perf_map;

        for (auto it : perfMap)
        {
            py::dict profile_info;
            switch (it.second.status)
            {
            case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
                profile_info["status"] = "EXECUTED";
                break;
            case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
                profile_info["status"] = "NOT_RUN";
                break;
            case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
                profile_info["status"] = "OPTIMIZED_OUT";
                break;
            default: profile_info["status"] = "UNKNOWN";
            }
            profile_info["exec_type"] = it.second.exec_type;
            profile_info["layer_type"] = it.second.layer_type;
            profile_info["cpu_time"] = it.second.cpu_uSec;
            profile_info["real_time"] = it.second.realTime_uSec;
            profile_info["execution_index"] = it.second.execution_index;
            perf_map[it.first.c_str()] = profile_info;
        }
        return perf_map;
    });

    cls.def("preprocess_info", &InferenceEngine::InferRequest::GetPreProcess, py::arg("name"));

    //    cls.def_property_readonly("preprocess_info", [](InferenceEngine::InferRequest& self) {
    //
    //    });
    //    cls.def_property_readonly("input_blobs", [](){
    //
    //    });
    //    cls.def_property_readonly("output_blobs", [](){
    //
    //    });

    //    latency
}
