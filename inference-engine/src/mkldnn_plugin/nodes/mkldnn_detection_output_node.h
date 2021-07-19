// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include "common/permute_kernel.h"

namespace MKLDNNPlugin {

class MKLDNNDetectionOutputNode : public MKLDNNNode {
public:
    MKLDNNDetectionOutputNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    const int idx_location = 0;
    const int idx_confidence = 1;
    const int idx_priors = 2;
    const int idx_arm_confidence = 3;
    const int idx_arm_location = 4;

    int _num_classes = 0;
    int _background_label_id = 0;
    int _top_k = 0;
    int _variance_encoded_in_target = 0;
    int _keep_top_k = 0;
    int _code_type = 0;

    bool _share_location    = false;
    bool _clip_before_nms   = false;  // clip bounding boxes before nms step
    bool _clip_after_nms    = false;  // clip bounding boxes after nms step
    bool _decrease_label_id = false;

    bool with_add_box_pred = false;

    int _image_width = 0;
    int _image_height = 0;
    int _prior_size = 4;
    bool _normalized = true;
    int _offset = 0;

    float _nms_threshold = 0.0f;
    float _confidence_threshold = 0.0f;
    float _objectness_score = 0.0f;

    int _batch_num = 0;
    int _num_loc_classes = 0;
    int _num_priors = 0;
    bool _priors_batches = false;
    int _conf_info_len = 0;

    enum CodeType {
        CORNER = 1,
        CENTER_SIZE = 2,
    };

    void getActualPriorNum(const float *prior_data, int* num_priors_actual, int n);

    void decodeBBoxes(const float *prior_data, const float *loc_data, const float *variance_data,
                      float *decoded_bboxes, float *decoded_bbox_sizes, int* num_priors_actual, int n, const int& offs, const int& pr_size,
                      bool decodeType = true, const int *conf_info_h = nullptr, const int *conf_info_v = nullptr); // decodeType is false after ARM

    void nms_cf(int* indicesIn, int& detections, int* indicesOut,
        const float* bboxes, const float* boxSizes);

    void nms_mx(int* indicesIn, int* detections, int* indicesOut,
        const float* bboxes, const float* sizes);

    void topk(const int *indicesIn, int *indicesOut, const float *conf, int n, int k);

    std::vector<float> _decoded_bboxes;
    std::vector<int> _buffer;
    std::vector<int> _indices;
    std::vector<int> _detections_count;
    std::vector<float> _reordered_conf;
    std::vector<float> _bbox_sizes;
    std::vector<int> _num_priors_actual;
    std::vector<int> _conf_info_prior;

    std::string errorPrefix;

    std::unique_ptr<PermuteKernel> permuteKernel_;
};

}  // namespace MKLDNNPlugin
