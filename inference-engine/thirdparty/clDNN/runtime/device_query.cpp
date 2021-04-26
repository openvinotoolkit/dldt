// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn/runtime/device_query.hpp"
#include "ocl/ocl_device_detector.hpp"

#ifdef CLDNN_WITH_SYCL
#include "sycl/sycl_device_detector.hpp"
#endif

#include <map>
#include <string>
namespace cldnn {

// We use runtime_type to filter out the same device with different execution runtime in order to skip it in devices list
// So we can have 2 logical devices for single physical GPU for OCL and L0 runtimes
// but user always see it as single device (e.g. GPU or GPU.0), and the actual runtime is specified in plugin config.
// Need to make sure that this is a good way to handle different backends from the users perspective and
// ensure that correct physical device is always selected for L0 case.
device_query::device_query(engine_types engine_type, runtime_types runtime_type, void* user_context, void* user_device) {
    switch (engine_type) {
#ifdef CLDNN_WITH_SYCL
    case engine_types::sycl: {
        sycl::sycl_device_detector sycl_detector;
        auto sycl_devices = sycl_detector.get_available_devices(runtime_type, user_context, user_device);
        _available_devices.insert(sycl_devices.begin(), sycl_devices.end());
        break;
    }
#endif
    case engine_types::ocl: {
        if (runtime_type != runtime_types::ocl)
            throw std::runtime_error("Unsupported runtime type for ocl engine");

        ocl::ocl_device_detector ocl_detector;
        _available_devices = ocl_detector.get_available_devices(user_context, user_device);
        break;
    }
    default: throw std::runtime_error("Unsupported engine type in device_query");
    }

    if (_available_devices.empty()) {
        throw std::runtime_error("No suitable devices found for requested engine and runtime types");
    }
}
}  // namespace cldnn
