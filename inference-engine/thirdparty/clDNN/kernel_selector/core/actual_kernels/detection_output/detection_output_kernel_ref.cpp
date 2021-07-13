// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detection_output_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey DetectionOutputKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants DetectionOutputKernelRef::GetJitConstants(const detection_output_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& detectOutParams = params.detectOutParams;

    jit.AddConstants({
        MakeJitConstant("NUM_IMAGES", detectOutParams.num_images),
        MakeJitConstant("NUM_CLASSES", detectOutParams.num_classes),
        MakeJitConstant("NUM_CLASSES_PER_ITEM", 4),
        MakeJitConstant("KEEP_TOP_K", detectOutParams.keep_top_k),
        MakeJitConstant("TOP_K", detectOutParams.top_k),
        MakeJitConstant("BACKGROUND_LABEL_ID", detectOutParams.background_label_id),
        MakeJitConstant("CODE_TYPE", detectOutParams.code_type),
        MakeJitConstant("CONF_SIZE_X", detectOutParams.conf_size_x),
        MakeJitConstant("CONF_SIZE_Y", detectOutParams.conf_size_y),
        MakeJitConstant("CONF_PADDING_X", detectOutParams.conf_padding_x),
        MakeJitConstant("CONF_PADDING_Y", detectOutParams.conf_padding_y),
        MakeJitConstant("SHARE_LOCATION", detectOutParams.share_location),
        MakeJitConstant("VARIANCE_ENCODED_IN_TARGET", detectOutParams.variance_encoded_in_target),
        MakeJitConstant("NMS_THRESHOLD", detectOutParams.nms_threshold),
        MakeJitConstant("ETA", detectOutParams.eta),
        MakeJitConstant("CONFIDENCE_THRESHOLD", detectOutParams.confidence_threshold),
        MakeJitConstant("IMAGE_WIDTH", detectOutParams.input_width),
        MakeJitConstant("IMAGE_HEIGH", detectOutParams.input_heigh),
        MakeJitConstant("DECREASE_LABEL_ID", detectOutParams.decrease_label_id),
        MakeJitConstant("CLIP_BEFORE_NMS", detectOutParams.clip_before_nms),
        MakeJitConstant("CLIP_AFTER_NMS", detectOutParams.clip_after_nms),
        MakeJitConstant("ELEMENTS_PER_THREAD", detectOutParams.elements_per_thread),
        MakeJitConstant("PRIOR_COORD_OFFSET", detectOutParams.prior_coordinates_offset),
        MakeJitConstant("PRIOR_INFO_SIZE", detectOutParams.prior_info_size),
        MakeJitConstant("PRIOR_IS_NORMALIZED", detectOutParams.prior_is_normalized),
    });

    return jit;
}

int GetPartitionStep(int localWorkItemNum) {
    int step_size = 0;
    for (int temp = localWorkItemNum; temp > 1; temp /= 2) {
        step_size++;
    }
    return step_size;
}

size_t GetOptimalLocalClassSize(std::vector<size_t> gws, const EngineInfo& info) {
    const size_t optimal_values[] = {16, 8, 7, 6, 5, 4, 2, 1};
    const size_t splitNum = gws[2];
    const size_t globalClassNum = gws[1];
    const auto rest_lws = info.maxWorkGroupSize / splitNum;
    size_t lws_idx = 0;
    while (rest_lws < optimal_values[lws_idx]) lws_idx++;
    while (globalClassNum % optimal_values[lws_idx]) lws_idx++;

    return optimal_values[lws_idx];
}

DetectionOutputKernelRef::DispatchData SetDefault(const detection_output_params& params, int idx) {
    DetectionOutputKernelRef::DispatchData dispatchData;
    const auto& input = params.inputs[0];
    const auto& detectOutParams = params.detectOutParams;
    constexpr size_t prior_box_size = 4;
    auto loc_feature_num = params.inputs[0].Feature().v;
    auto num_classes = detectOutParams.num_classes;
    auto num_loc_classes = (detectOutParams.share_location) ? 1 : num_classes;
    auto num_prior_boxes = (loc_feature_num / (num_loc_classes * prior_box_size));

    if (idx == 0) {
        if (detectOutParams.decrease_label_id) {
            dispatchData.gws = {input.Batch().v, num_prior_boxes, 1};
            dispatchData.lws = {1, 1, 1};
        } else {
            if (detectOutParams.conf_padding_x || detectOutParams.conf_padding_y) {
                dispatchData.gws = {num_classes, 256, input.Batch().v};
            } else {
                dispatchData.gws = {CeilDiv(num_classes, 4), 256, input.Batch().v};
            }
            dispatchData.lws = {1, 256, 1};
        }
    } else if (idx == 1) {
        const size_t kSplitNum = 16;
        if (detectOutParams.decrease_label_id) {
            dispatchData.gws = {input.Batch().v, 1, kSplitNum};
            dispatchData.lws = {1, 1, kSplitNum};
        } else {
            dispatchData.gws = {input.Batch().v, num_classes, kSplitNum};
            const size_t kClassSize = GetOptimalLocalClassSize(dispatchData.gws, params.engineInfo);
            dispatchData.lws = {1, kClassSize, kSplitNum};
        }
    } else if (idx == 2) {
        if (detectOutParams.decrease_label_id) {
            dispatchData.gws = {input.Batch().v, 1, 1};
            dispatchData.lws = {1, 1, 1};
        } else {
            dispatchData.gws = {input.Batch().v, num_classes, 1};
            dispatchData.lws = {1, 1, 1};
        }
    } else if (idx == 3) {
        if (detectOutParams.decrease_label_id) {
            dispatchData.gws = {1, 1, 1};
            dispatchData.lws = {1, 1, 1};
        } else {
            dispatchData.gws = {input.Batch().v, 1, 1};
            dispatchData.lws = {1, 1, 1};
        }
    } else {
        dispatchData.gws = {1, 1, 1};
        dispatchData.lws = {1, 1, 1};
    }

    return dispatchData;
}

void DetectionOutputKernelRef::SetKernelArguments(const detection_output_params& params, clKernelData& kernel, size_t idx) const {
    if (idx == 0) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
    } else if (idx == 1) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
    } else if (idx == 2) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
    } else if (idx == 3) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
    }
}

KernelsData DetectionOutputKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::DETECTION_OUTPUT && options.GetType() == KernelType::DETECTION_OUTPUT);

    constexpr size_t kKernelsNum = 4;
    KernelData kd = KernelData::Default<detection_output_params>(params, kKernelsNum);
    const detection_output_params& detectOutParams = static_cast<const detection_output_params&>(params);

    constexpr size_t prior_box_size = 4;
    auto num_of_images = detectOutParams.inputs[0].Batch().v;
    auto loc_feature_num = detectOutParams.inputs[0].Feature().v;
    auto num_classes = detectOutParams.detectOutParams.num_classes;
    auto num_loc_classes = (detectOutParams.detectOutParams.share_location) ? 1 : num_classes;
    auto num_prior_boxes = (loc_feature_num / (num_loc_classes * prior_box_size));

    constexpr size_t buffer_bytes = 16;  // The size of struct Scores in detection_output_gpu_ref.cl
    size_t buffer_stride = num_prior_boxes * buffer_bytes;
    size_t buffer_size = num_of_images * num_classes * buffer_stride;
    size_t num_scores_size = num_of_images * (num_classes + 1) * sizeof(int);

    kd.internalBufferSizes.push_back(buffer_size);
    kd.internalBufferSizes.push_back(buffer_size);
    kd.internalBufferSizes.push_back(num_scores_size);
    kd.internalBufferDataType = GetUnitType(detectOutParams);

    for (size_t i = 0; i < kKernelsNum; i++) {
        DispatchData dispatchData = SetDefault(detectOutParams, i);
        auto cldnnJit = GetJitConstants(detectOutParams);
        auto entryPoint = GetEntryPoint(kernelName, detectOutParams.layerID, options);
        cldnnJit.AddConstant(MakeJitConstant("BUFFER_STRIDE", buffer_stride));
        if (i == 0) {
            if (detectOutParams.detectOutParams.decrease_label_id) {
                cldnnJit.AddConstant(MakeJitConstant("DO_STAGE_" + std::to_string(i) + "_MXNET", "true"));
            } else {
                if (detectOutParams.detectOutParams.conf_padding_x || detectOutParams.detectOutParams.conf_padding_y) {
                    cldnnJit.AddConstants({MakeJitConstant("DO_STAGE_" + std::to_string(i) + "_CAFFE", "true")});
                } else {
                    cldnnJit.AddConstants({MakeJitConstant("DO_STAGE_" + std::to_string(i) + "_CAFFE_OPT", "true")});
                }
                size_t num_bit_mask = CeilDiv(num_prior_boxes, 8);
                size_t num_score_per_item = RoundUp(CeilDiv(num_prior_boxes, 256), 8);
                size_t num_score_block = CeilDiv(num_prior_boxes, num_score_per_item);
                cldnnJit.AddConstants({MakeJitConstant("NUM_BIT_MASK", num_bit_mask),
                                       MakeJitConstant("NUM_PRIORS_PER_ITEM", num_score_per_item),
                                       MakeJitConstant("NUM_PRIOR_BLOCKS", num_score_block)});
            }
        } else if (i == 1) {
             if (detectOutParams.detectOutParams.decrease_label_id) {
                cldnnJit.AddConstants({MakeJitConstant("DO_STAGE_" + std::to_string(i) + "_MXNET", "true"),
                                       MakeJitConstant("LOCAL_WORK_NUM", dispatchData.lws[2]),
                                       MakeJitConstant("PARTITION_STEP", GetPartitionStep(dispatchData.lws[2]))});
             } else {
                cldnnJit.AddConstants({MakeJitConstant("DO_STAGE_" + std::to_string(i) + "_CAFFE", "true"),
                                       MakeJitConstant("LOCAL_CLASS_NUM", dispatchData.lws[1]),
                                       MakeJitConstant("LOCAL_WORK_NUM", dispatchData.lws[2]),
                                       MakeJitConstant("PARTITION_STEP", GetPartitionStep(dispatchData.lws[2]))});
             }
         } else if (i == 2) {
            if (detectOutParams.detectOutParams.decrease_label_id) {
                cldnnJit.AddConstant(MakeJitConstant("DO_STAGE_" + std::to_string(i) + "_MXNET", "true"));
            } else {
                if (detectOutParams.detectOutParams.top_k > 0) {
                    cldnnJit.AddConstant(MakeJitConstant("DO_STAGE_" + std::to_string(i) + "_CAFFE_OPT", "true"));
                } else {
                    cldnnJit.AddConstant(MakeJitConstant("DO_STAGE_" + std::to_string(i) + "_CAFFE", "true"));
                }
            }
         } else {
            if (detectOutParams.detectOutParams.decrease_label_id) {
                cldnnJit.AddConstant(MakeJitConstant("DO_STAGE_" + std::to_string(i) + "_MXNET", "true"));
            } else {
                cldnnJit.AddConstant(MakeJitConstant("DO_STAGE_" + std::to_string(i) + "_CAFFE", "true"));
            }
        }

        auto jit = CreateJit(kernelName, cldnnJit, entryPoint);
        auto& kernel = kd.kernels[i];
        KernelBase::CheckDispatchData(kernelName, dispatchData);
        kernel.params.workGroups.global = dispatchData.gws;
        kernel.params.workGroups.local  = dispatchData.lws;
        kernel.code.kernelString = GetKernelString(kernelName, jit, entryPoint, params.engineInfo);
        SetKernelArguments(detectOutParams, kernel, i);
    }

    return {kd};
}

KernelsPriority DetectionOutputKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
