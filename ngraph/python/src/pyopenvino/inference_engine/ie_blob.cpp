//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <ie_blob.h>
#include <ie_common.h>
#include <ie_layouts.h>
#include <ie_precision.hpp>

#include <pybind11/stl.h>
#include "../../../pybind11/include/pybind11/pybind11.h"
#include "ie_blob.h"
#include "pyopenvino/inference_engine/ie_blob.hpp"
#include "pyopenvino/inference_engine/tensor_description.hpp"

namespace py = pybind11;

void regclass_Blob(py::module m) {
    py::class_<InferenceEngine::Blob, std::shared_ptr<InferenceEngine::Blob>> cls(m, "Blob");
}
