// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the Inference Engine plugins destruction mechanism
 * 
 * @file ie_irelease.hpp
 */
#pragma once

#include <memory>

#include "ie_api.h"
#include "ie_no_copy.hpp"

namespace InferenceEngine {
namespace details {
/**
 * @brief This class is used for objects allocated by a shared module (in *.so)
 */
// IE_SUPPRESS_DEPRECATED_START
// INFERENCE_ENGINE_DEPRECATED("This Interface is depricated")
class IRelease : public no_copy {
public:
    /**
     * @brief Releases current allocated object and all related resources.
     * Once this method is called, the pointer to this interface is no longer valid
     */
    virtual void Release() noexcept {
        delete this;
    }

protected:
    /**
     * @brief Default destructor
     */
    ~IRelease() override = default;
};
// IE_SUPPRESS_DEPRECATED_END
}  // namespace details
}  // namespace InferenceEngine
