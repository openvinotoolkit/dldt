// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "extract_image_patches_kernel_base.h"

namespace kernel_selector {
class ExtractImagePatchesKernelRef : public ExtractImagePatchesKernelBase {
public:
    ExtractImagePatchesKernelRef() : ExtractImagePatchesKernelBase("extract_image_patches_ref") {}
    virtual ~ExtractImagePatchesKernelRef() = default;
protected:
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
};
}  // namespace kernel_selector
