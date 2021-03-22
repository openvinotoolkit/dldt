// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include <cpp/ie_infer_request.hpp>

namespace py = pybind11;

void regclass_InferRequest(py::module m);
