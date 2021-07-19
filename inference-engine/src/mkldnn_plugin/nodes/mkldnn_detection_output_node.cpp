// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "base.hpp"

#include <string>
#include <vector>
#include <mutex>

#include <ngraph/op/detection_output.hpp>
#include "ie_parallel.hpp"
#include "mkldnn_detection_output_node.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

template <typename T>
bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2) {
    return (pair1.first > pair2.first) || (pair1.first == pair2.first && pair1.second < pair2.second);
}

template <>
bool SortScorePairDescend<std::pair<int, int>>(const std::pair<float, std::pair<int, int>>& pair1,
                                               const std::pair<float, std::pair<int, int>>& pair2) {
    return (pair1.first > pair2.first) || (pair1.first == pair2.first && pair1.second.second < pair2.second.second);
}

bool MKLDNNDetectionOutputNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto doOp = ngraph::as_type_ptr<const ngraph::op::v0::DetectionOutput>(op);
        if (!doOp) {
            errorMessage = "Node is not an instance of the DetectionOutput from the operations set v0.";
            return false;
        }
        if (!details::CaselessEq<std::string>()(doOp->get_attrs().code_type, "caffe.PriorBoxParameter.CENTER_SIZE") &&
            !details::CaselessEq<std::string>()(doOp->get_attrs().code_type, "caffe.PriorBoxParameter.CORNER")) {
            errorMessage = "Unsupported code_type attribute: " + doOp->get_attrs().code_type;
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNDetectionOutputNode::MKLDNNDetectionOutputNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache), permuteKernel_(nullptr) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "DetectionOutput layer with name '" + op->get_friendly_name() + "' ";

    if (getOriginalInputsNumber() != 3 && getOriginalInputsNumber() != 5)
        IE_THROW() << errorPrefix <<  " has incorrect number of input edges.";

    if (getOriginalOutputsNumber() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of output edges.";

    auto doOp = ngraph::as_type_ptr<const ngraph::op::v0::DetectionOutput>(op);
    auto attributes = doOp->get_attrs();

    _num_classes = attributes.num_classes;
    _background_label_id = attributes.background_label_id;
    _top_k = attributes.top_k;
    _variance_encoded_in_target = attributes.variance_encoded_in_target;
    _keep_top_k = attributes.keep_top_k[0];
    _nms_threshold = attributes.nms_threshold;
    _confidence_threshold = attributes.confidence_threshold;
    _share_location = attributes.share_location;
    _clip_before_nms = attributes.clip_before_nms;
    _clip_after_nms = attributes.clip_after_nms;
    _decrease_label_id = attributes.decrease_label_id;
    _normalized = attributes.normalized;
    _image_height = attributes.input_height;
    _image_width = attributes.input_width;
    _prior_size = _normalized ? 4 : 5;
    _offset = _normalized ? 0 : 1;
    _num_loc_classes = _share_location ? 1 : _num_classes;

    with_add_box_pred = getOriginalInputsNumber() == 5;
    _objectness_score = attributes.objectness_score;

    _code_type = (details::CaselessEq<std::string>()(attributes.code_type, "caffe.PriorBoxParameter.CENTER_SIZE") ?
                  CodeType::CENTER_SIZE : CodeType::CORNER);

    _num_priors = static_cast<int>(op->get_input_shape(idx_priors).back() / _prior_size);
    _priors_batches = op->get_input_shape(idx_priors).front() != 1;

    if (_num_priors * _num_loc_classes * 4 != static_cast<int>(op->get_input_shape(idx_location)[1]))
        IE_THROW() << errorPrefix << " has incorrect number of priors must match number of location predictions ("
                   << _num_priors * _num_loc_classes * 4 << " vs "
                   << op->get_input_shape(idx_location)[1] << ")";

    if (_num_priors * _num_classes != static_cast<int>(op->get_input_shape(idx_confidence).back()))
        IE_THROW() << " has incorrect number of priors must match number of confidence predictions.";

    if (_decrease_label_id && _background_label_id != 0)
        IE_THROW() << errorPrefix << " cannot use decrease_label_id and background_label_id parameter simultaneously.";

    _batch_num = static_cast<int>(op->get_input_shape(idx_confidence)[0]);

    _decoded_bboxes.resize(_batch_num * _num_classes * _num_priors * 4);
    _bbox_sizes.resize(_batch_num * _num_classes * _num_priors);
    _buffer.resize(_batch_num * _num_classes * _num_priors);
    _indices.resize(_batch_num * _num_classes * _num_priors);
    // for shared_location
    _conf_info_prior.resize(_batch_num * _num_priors);

    // confs...count...indices, caffe style and filter case.
    // caffe: conf_info for sparsity or indices(filter) --> topk(buffer) --> nms(indices) --> g_topk(vector<>(all detections) --> indices per class))
    // MXNet: indices(filter), data if for image --> topk(buffer) --> nms(indices) --> g_topk(vector<>(all detections) --> indices per class))
    _conf_info_len = (!_decrease_label_id && (_confidence_threshold != 0.0f)) ? (2 * _num_priors + 1) : _num_priors;
    _reordered_conf.resize(_batch_num * _num_classes * _conf_info_len);

    _detections_count.resize(_batch_num * _num_classes);
    _num_priors_actual.resize(_batch_num);
}

void MKLDNNDetectionOutputNode::createPrimitive() {
    PermuteParams params;
    params.src_block_dims = {static_cast<size_t>(_batch_num), static_cast<size_t>(_num_priors), static_cast<size_t>(_num_classes)};
    params.dst_block_dims = {static_cast<size_t>(_batch_num), static_cast<size_t>(_num_classes), static_cast<size_t>(_num_priors)};
    params.order = {0, 2, 1};
    params.src_block_order = {0, 1, 2};
    params.dst_block_order = {0, 1, 2};
    params.data_size = sizeof(float);
    permuteKernel_ = std::unique_ptr<PermuteKernel>(new PermuteKernel(params));
}

void MKLDNNDetectionOutputNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<DataConfigurator> inDataConf;
    inDataConf.reserve(getOriginalInputsNumber());
    for (int i = 0; i < getOriginalInputsNumber(); ++i)
        inDataConf.emplace_back(TensorDescCreatorTypes::ncsp, Precision::FP32);

    addSupportedPrimDesc(inDataConf,
                         {{TensorDescCreatorTypes::ncsp, Precision::FP32}},
                         impl_desc_type::ref_any);
}

void MKLDNNDetectionOutputNode::getActualPriorNum(const float *prior_data, int* num_priors_actual, int n) {
    num_priors_actual[n] = _num_priors;
    if (!_normalized) {
        int num = 0;
        for (; num < _num_priors; ++num) {
            float batch_id = prior_data[num * _prior_size + 0];
            if (batch_id == -1.f) {
                num_priors_actual[n] = num;
                break;
            }
        }
    }
}

struct ConfidenceComparator {
    explicit ConfidenceComparator(const float* conf_data) : _conf_data(conf_data) {}

    bool operator()(int idx1, int idx2) {
        if (_conf_data[idx1] > _conf_data[idx2]) return true;
        if (_conf_data[idx1] < _conf_data[idx2]) return false;
        return idx1 < idx2;
    }

    const float* _conf_data;
};

void MKLDNNDetectionOutputNode::execute(mkldnn::stream strm) {
    float *dst_data = reinterpret_cast<float *>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());

    const float *loc_data    = reinterpret_cast<const float *>(getParentEdgeAt(idx_location)->getMemoryPtr()->GetPtr());
    const float *conf_data   = reinterpret_cast<const float *>(getParentEdgeAt(idx_confidence)->getMemoryPtr()->GetPtr());
    const float *prior_data  = reinterpret_cast<const float *>(getParentEdgeAt(idx_priors)->getMemoryPtr()->GetPtr());
    const float *arm_conf_data = inDims.size() > 3 ?
            reinterpret_cast<const float *>(getParentEdgeAt(idx_arm_confidence)->getMemoryPtr()->GetPtr()) : nullptr;
    const float *arm_loc_data = inDims.size() > 4 ?
            reinterpret_cast<const float *>(getParentEdgeAt(idx_arm_location)->getMemoryPtr()->GetPtr()) : nullptr;

    int batch_for_priors = _priors_batches ? _batch_num : 1;
    int *num_priors_actual = _num_priors_actual.data();
    for (int n = 0; n < batch_for_priors; ++n) {
        const float *ppriors = prior_data;
        ppriors += _variance_encoded_in_target ? (n * _num_priors * _prior_size) : (2 * n * _num_priors * _prior_size);
        getActualPriorNum(ppriors, num_priors_actual, n);
    }
    if (!_priors_batches && _batch_num > 1) {
        for (int n = 1; n < _batch_num; ++n) {
            num_priors_actual[n] = num_priors_actual[0];
        }
    }

    float *decoded_bboxes_data = _decoded_bboxes.data();
    float *bbox_sizes_data     = _bbox_sizes.data();
    int *detections_data       = _detections_count.data();
    int *buffer_data           = _buffer.data();
    int *indices_data          = _indices.data();

    memset(detections_data, 0, _batch_num*_num_classes*sizeof(int));

    // confs extract and reorder
    float *reordered_conf_data = _reordered_conf.data();
    int *reordered_conf_data_indices = reinterpret_cast<int*>(_reordered_conf.data());
    if (_confidence_threshold == 0.0f) {
        if (with_add_box_pred) {
            for (int n = 0; n < _batch_num; ++n) {
                for (int p = 0; p < _num_priors; ++p) {
                    if (arm_conf_data[n*_num_priors*2 + p * 2 + 1] < _objectness_score) {
                        for (int c = 0; c < _num_classes; ++c) {
                            reordered_conf_data[n*_num_priors*_num_classes + c*_num_priors + p] = c == _background_label_id ? 1.0f : 0.0f;
                        }
                    } else {
                        for (int c = 0; c < _num_classes; ++c) {
                            reordered_conf_data[n*_num_priors*_num_classes + c*_num_priors + p] = conf_data[n*_num_priors*_num_classes + p*_num_classes + c];
                        }
                    }
                }
            }
        } else {
            if (permuteKernel_) {
                auto srcData = reinterpret_cast<const uint8_t*>(conf_data);
                auto dstData = reinterpret_cast<uint8_t*>(reordered_conf_data);
                permuteKernel_->execute(srcData, dstData);
            } else {
                for (int n = 0; n < _batch_num; ++n) {
                    for (int c = 0; c < _num_classes; ++c) {
                        for (int p = 0; p < _num_priors; ++p) {
                            reordered_conf_data[n*_num_priors*_num_classes + c*_num_priors + p] = conf_data[n*_num_priors*_num_classes + p*_num_classes + c];
                        }
                    }
                }
            }
        }

        // topk for Caffe
        if (!_decrease_label_id) {
            parallel_for2d(_batch_num, _num_classes, [&](size_t n, size_t c) {
                // in:  conf
                // out: detectionCount, buffer
                if (c == _background_label_id)
                    return;
                int off = n*_num_priors*_num_classes + c*_num_priors;
                const float *pconf = reordered_conf_data + off;
                int *pindices = indices_data + off;
                int *pbuffer = buffer_data + off;

                int count = 0;
                for (int i = 0; i < _num_priors_actual[n]; ++i) {
                    if (pconf[i] > _confidence_threshold) {
                        pindices[count] = i;
                        count++;
                    }
                }

                int k = (_top_k == -1 ? count : (std::min)(_top_k, count));
                topk(pindices, pbuffer, pconf, count, k);
                detections_data[n*_num_classes + c] = k;
            });
        } else { // topk for MXNet
            for (int n = 0; n < _batch_num; ++n) {
                int offB = n * _num_priors * _num_classes;
                std::mutex mtx;
                parallel_for(_num_priors_actual[n], [&](size_t p) {
                    // in:  origin conf
                    // out: detectionCount, buffer
                    bool arm_prior = false;
                    if (with_add_box_pred)
                        arm_prior = arm_conf_data[n*_num_priors*2 + p * 2 + 1] < _objectness_score;
                    float maxConf = -1;
                    int maxCIdx = 0;
                    for (int c = 1; c < _num_classes; ++c) {
                        float conf = conf_data[offB + p * _num_classes + c];
                        if (with_add_box_pred && arm_prior)
                            conf = (c == _background_label_id) ? 1.0f : 0.0f;  // still need refresh conf due to read from origin
                        if (conf >= _confidence_threshold && conf > maxConf) {
                            maxConf = conf;
                            maxCIdx = c;
                        }
                    }
                    if (maxCIdx > 0) {
                        mtx.lock();
                        indices_data[offB + detections_data[n*_num_classes]] = maxCIdx*_num_priors + p;  // de-refer to get prior and class id.
                        detections_data[n*_num_classes]++;
                        mtx.unlock();
                    }
                });

                // in: indices_data, detection_count(filtered num)
                // out: buffer, detection_count(k)
                int count = detections_data[n*_num_classes];
                int k = (_top_k == -1 ? count : (std::min)(_top_k, count));

                const float *pconf = reordered_conf_data + offB;
                int *indices = indices_data + offB;
                int *pbuffer = buffer_data + offB;
                topk(indices, pbuffer, pconf, count, k);
                detections_data[n*_num_classes] = k;
            }
        }
    } else {
        for (int n = 0; n < _batch_num; ++n) {
            int off = n * _num_priors * _num_classes;
            int offH = n * _conf_info_len * _num_classes; // horizontal info
            int offV = n * _num_priors;  // vertical info
            // reset count
            if (!_decrease_label_id) {
                parallel_for(_num_classes, [&](size_t c) {
                    int countIdx = offH + c * _conf_info_len + _num_priors;
                    reordered_conf_data_indices[countIdx] = 0;
                });
            }

            // reorder + build conf info
            std::mutex mtx;
            parallel_for(_num_priors_actual[n], [&](size_t p) {
                bool arm_prior = false;
                if (with_add_box_pred)
                    arm_prior = arm_conf_data[n*_num_priors*2 + p * 2 + 1] < _objectness_score;
                if (_share_location)
                    _conf_info_prior[offV + p] = -1;
                float maxConf = -1;
                int maxCIdx = 0;
                int confIdxPrior = off + p * _num_classes;
                for (int c = 0; c < _num_classes; ++c) {
                    if (!_decrease_label_id && c == _background_label_id)
                        continue;
                    float conf = conf_data[confIdxPrior + c];
                    if (with_add_box_pred && arm_prior)
                        conf = (c == _background_label_id) ? 1.0f : 0.0f;
                    if (conf >= _confidence_threshold) {
                        if (!_decrease_label_id && conf == _confidence_threshold)
                            continue;
                        int idx = offH + c * _conf_info_len;
                        reordered_conf_data[idx + p] = conf;
                        if (!_decrease_label_id) {
                            mtx.lock();
                            reordered_conf_data_indices[idx + _num_priors]++;
                            reordered_conf_data_indices[idx + _num_priors + reordered_conf_data_indices[idx + _num_priors]] = p;
                            mtx.unlock();
                        }

                        // vertical info for _share_location(flag to decode for each prior)
                        if (_share_location) {
                            _conf_info_prior[offV + p] = 1; // 1 for decode
                        }
                        // vertical info for MXNet style(max conf for each prior)
                        if (_decrease_label_id && c != 0) {
                            if (conf > maxConf) {
                                maxConf = conf;
                                maxCIdx = c;
                            }
                        }
                    }
                }
                // MXNet statistic, detection_count and indices is for each image
                if (_decrease_label_id && maxCIdx > 0) {
                    mtx.lock();
                    indices_data[off + detections_data[n*_num_classes]] = maxCIdx*_num_priors + p;  // de-refer to get prior and class id.
                    detections_data[n*_num_classes]++;
                    mtx.unlock();
                }
            });

            // topk for Caffe style
            if (!_decrease_label_id) {
                parallel_for(_num_classes, [&](size_t c) {
                    // in:  conf_h info
                    // out: detectionCount, buffer
                    int countIdx = offH + c * _conf_info_len + _num_priors;
                    int count = reordered_conf_data_indices[countIdx];
                    int k = (_top_k == -1 ? count : (std::min)(_top_k, count));

                    int *reordered_conf_indices = reordered_conf_data_indices + countIdx + 1;
                    int *pbuffer = buffer_data + off + c*_num_priors;
                    const float *pconf = reordered_conf_data + offH + c*_conf_info_len;

                    topk(reordered_conf_indices, pbuffer, pconf, count, k);
                    detections_data[n*_num_classes + c] = k;
                });
            } else { // topk MXNet style
                // in: indices_data, detection_count(filtered num)
                // out: buffer, detection_count(k)
                int count = detections_data[n*_num_classes];
                int k = (_top_k == -1 ? count : (std::min)(_top_k, count));

                const float *pconf = reordered_conf_data + off;
                int *indices = indices_data + off;
                int *pbuffer = buffer_data + off;
                topk(indices, pbuffer, pconf, count, k);
                detections_data[n*_num_classes] = k;
            }
        }
    }

    int *conf_info_v = _conf_info_prior.data();

    for (int n = 0; n < _batch_num; ++n) {
        const float *ppriors = prior_data;
        const float *prior_variances = prior_data + _num_priors*_prior_size;
        if (_priors_batches) {
            ppriors += _variance_encoded_in_target ? n*_num_priors*_prior_size : 2*n*_num_priors*_prior_size;
            prior_variances += _variance_encoded_in_target ? 0 : 2*n*_num_priors*_prior_size;
        }

        if (_share_location) {
            const float *ploc = loc_data + n*4*_num_priors;
            float *pboxes = decoded_bboxes_data + n*4*_num_priors;
            float *psizes = bbox_sizes_data + n*_num_priors;
            int *conf_info_v_b = conf_info_v + n * _num_priors;

            if (with_add_box_pred) {
                const float *p_arm_loc = arm_loc_data + n*4*_num_priors;
                decodeBBoxes(ppriors, p_arm_loc, prior_variances, pboxes, psizes, num_priors_actual, n, _offset, _prior_size, true, nullptr, conf_info_v_b);
                decodeBBoxes(pboxes, ploc, prior_variances, pboxes, psizes, num_priors_actual, n, 0, 4, false, nullptr, conf_info_v_b);
            } else {
                decodeBBoxes(ppriors, ploc, prior_variances, pboxes, psizes, num_priors_actual, n, _offset, _prior_size, true, nullptr, conf_info_v_b);
            }
        } else {
            for (int c = 0; c < _num_loc_classes; ++c) {
                if (c == _background_label_id) {
                    continue;
                }
                const float *ploc = loc_data + n*4*_num_loc_classes*_num_priors + c*4;
                float *pboxes = decoded_bboxes_data + n*4*_num_loc_classes*_num_priors + c*4*_num_priors;
                float *psizes = bbox_sizes_data + n*_num_loc_classes*_num_priors + c*_num_priors;
                int *conf_info_h_bc = reordered_conf_data_indices + n * _conf_info_len * _num_classes + c*_conf_info_len;
                if (with_add_box_pred) {
                    const float *p_arm_loc = arm_loc_data + n*4*_num_loc_classes*_num_priors + c*4;
                    decodeBBoxes(ppriors, p_arm_loc, prior_variances, pboxes, psizes, num_priors_actual, n, _offset, _prior_size, true, conf_info_h_bc);
                    decodeBBoxes(pboxes, ploc, prior_variances, pboxes, psizes, num_priors_actual, n, 0, 4, false, conf_info_h_bc);
                } else {
                    decodeBBoxes(ppriors, ploc, prior_variances, pboxes, psizes, num_priors_actual, n, _offset, _prior_size, true, conf_info_h_bc);
                }
            }
        }
    }

    for (int n = 0; n < _batch_num; ++n) {
        int detections_total = 0;

        if (!_decrease_label_id) {
            // Caffe style
            parallel_for(_num_classes, [&](int c) {
                if (c != _background_label_id) {  // Ignore background class
                    int *pindices    = indices_data + n*_num_classes*_num_priors + c*_num_priors;
                    int *pbuffer     = buffer_data + n*_num_classes*_num_priors + c*_num_priors;
                    int *pdetections = detections_data + n*_num_classes + c;

                    const float *pboxes;
                    const float *psizes;
                    if (_share_location) {
                        pboxes = decoded_bboxes_data + n*4*_num_priors;
                        psizes = bbox_sizes_data + n*_num_priors;
                    } else {
                        pboxes = decoded_bboxes_data + n*4*_num_classes*_num_priors + c*4*_num_priors;
                        psizes = bbox_sizes_data + n*_num_classes*_num_priors + c*_num_priors;
                    }

                    nms_cf(pbuffer, *pdetections, pindices, pboxes, psizes);
                }
            });
        } else {
            // MXNet style
            int *pbuffer = buffer_data + n*_num_classes*_num_priors;
            int *pdetections = detections_data + n*_num_classes;
            int *pindices = indices_data + n*_num_classes*_num_priors;
            const float *pboxes = decoded_bboxes_data + n*4*_num_loc_classes*_num_priors;
            const float *psizes = bbox_sizes_data + n*_num_loc_classes*_num_priors;

            nms_mx(pbuffer, pdetections, pindices, pboxes, psizes);
        }

        for (int c = 0; c < _num_classes; ++c) {
            detections_total += detections_data[n*_num_classes + c];
        }

        // combine detections of all class for this image and filter with global(image) topk(keep_topk)
        if (_keep_top_k > -1 && detections_total > _keep_top_k) {
            std::vector<std::pair<float, std::pair<int, int>>> conf_index_class_map;

            for (int c = 0; c < _num_classes; ++c) {
                int detections = detections_data[n*_num_classes + c];
                int *pindices = indices_data + n*_num_classes*_num_priors + c*_num_priors;

                float *pconf  = reordered_conf_data + n*_num_classes*_conf_info_len + c*_conf_info_len;

                for (int i = 0; i < detections; ++i) {
                    int idx = pindices[i];
                    conf_index_class_map.push_back(std::make_pair(pconf[idx], std::make_pair(c, idx)));
                }
            }

            std::sort(conf_index_class_map.begin(), conf_index_class_map.end(),
                      SortScorePairDescend<std::pair<int, int>>);
            conf_index_class_map.resize(_keep_top_k);

            // Store the new indices.
            memset(detections_data + n*_num_classes, 0, _num_classes * sizeof(int));

            for (size_t j = 0; j < conf_index_class_map.size(); ++j) {
                int label = conf_index_class_map[j].second.first;
                int idx = conf_index_class_map[j].second.second;
                int *pindices = indices_data + n * _num_classes * _num_priors + label * _num_priors;
                pindices[detections_data[n*_num_classes + label]] = idx;
                detections_data[n*_num_classes + label]++;
            }
        }
    }

    const int num_results = getChildEdgesAtPort(0)[0]->getDims()[2];
    const int DETECTION_SIZE = getChildEdgesAtPort(0)[0]->getDims()[3];
    if (DETECTION_SIZE != 7) {
        IE_THROW() << NOT_IMPLEMENTED;
    }

    int dst_data_size = 0;
    if (_keep_top_k > 0)
        dst_data_size = _batch_num * _keep_top_k * DETECTION_SIZE * sizeof(float);
    else if (_top_k > 0)
        dst_data_size = _batch_num * _top_k * _num_classes * DETECTION_SIZE * sizeof(float);
    else
        dst_data_size = _batch_num * _num_classes * _num_priors * DETECTION_SIZE * sizeof(float);

    if (dst_data_size > getChildEdgesAtPort(0)[0]->getBlob()->byteSize()) {
        IE_THROW() << OUT_OF_BOUNDS;
    }
    memset(dst_data, 0, dst_data_size);

    // set final detection result to output blob
    int count = 0;
    for (int n = 0; n < _batch_num; ++n) {
        const float *pconf   = reordered_conf_data + n * _conf_info_len * _num_classes;
        const float *pboxes  = decoded_bboxes_data + n*_num_priors*4*_num_loc_classes;
        const int *pindices  = indices_data + n*_num_classes*_num_priors;

        for (int c = 0; c < _num_classes; ++c) {
            for (int i = 0; i < detections_data[n*_num_classes + c]; ++i) {
                int prIdx = pindices[c*_num_priors + i];

                dst_data[count * DETECTION_SIZE + 0] = static_cast<float>(n);
                dst_data[count * DETECTION_SIZE + 1] = static_cast<float>(_decrease_label_id ? c-1 : c);
                dst_data[count * DETECTION_SIZE + 2] = pconf[c*_conf_info_len + prIdx];

                float xmin = _share_location ? pboxes[prIdx*4 + 0] :
                             pboxes[c*4*_num_priors + prIdx*4 + 0];
                float ymin = _share_location ? pboxes[prIdx*4 + 1] :
                             pboxes[c*4*_num_priors + prIdx*4 + 1];
                float xmax = _share_location ? pboxes[prIdx*4 + 2] :
                             pboxes[c*4*_num_priors + prIdx*4 + 2];
                float ymax = _share_location ? pboxes[prIdx*4 + 3] :
                             pboxes[c*4*_num_priors + prIdx*4 + 3];

                if (_clip_after_nms) {
                    xmin = (std::max)(0.0f, (std::min)(1.0f, xmin));
                    ymin = (std::max)(0.0f, (std::min)(1.0f, ymin));
                    xmax = (std::max)(0.0f, (std::min)(1.0f, xmax));
                    ymax = (std::max)(0.0f, (std::min)(1.0f, ymax));
                }

                dst_data[count * DETECTION_SIZE + 3] = xmin;
                dst_data[count * DETECTION_SIZE + 4] = ymin;
                dst_data[count * DETECTION_SIZE + 5] = xmax;
                dst_data[count * DETECTION_SIZE + 6] = ymax;

                ++count;
            }
        }
    }

    if (count < num_results) {
        // marker at end of boxes list
        dst_data[count * DETECTION_SIZE + 0] = -1;
    }
}

static inline float JaccardOverlap(const float *decoded_bbox,
                                   const float *bbox_sizes,
                                   const int idx1,
                                   const int idx2) {
    float xmin1 = decoded_bbox[idx1*4 + 0];
    float ymin1 = decoded_bbox[idx1*4 + 1];
    float xmax1 = decoded_bbox[idx1*4 + 2];
    float ymax1 = decoded_bbox[idx1*4 + 3];

    float xmin2 = decoded_bbox[idx2*4 + 0];
    float ymin2 = decoded_bbox[idx2*4 + 1];
    float xmax2 = decoded_bbox[idx2*4 + 2];
    float ymax2 = decoded_bbox[idx2*4 + 3];

    if (xmin2 > xmax1 || xmax2 < xmin1 || ymin2 > ymax1 || ymax2 < ymin1) {
        return 0.0f;
    }

    float intersect_xmin = (std::max)(xmin1, xmin2);
    float intersect_ymin = (std::max)(ymin1, ymin2);
    float intersect_xmax = (std::min)(xmax1, xmax2);
    float intersect_ymax = (std::min)(ymax1, ymax2);

    float intersect_width  = intersect_xmax - intersect_xmin;
    float intersect_height = intersect_ymax - intersect_ymin;

    if (intersect_width <= 0 || intersect_height <= 0) {
        return 0.0f;
    }

    float intersect_size = intersect_width * intersect_height;
    float bbox1_size = bbox_sizes[idx1];
    float bbox2_size = bbox_sizes[idx2];

    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
}

void MKLDNNDetectionOutputNode::decodeBBoxes(const float *prior_data,
                                       const float *loc_data,
                                       const float *variance_data,
                                       float *decoded_bboxes,
                                       float *decoded_bbox_sizes,
                                       int* num_priors_actual,
                                       int n,
                                       const int& offs,
                                       const int& pr_size,
                                       bool decodeType,
                                       const int *conf_info_h,
                                       const int *conf_info_v) {
    int prNum = num_priors_actual[n];
    if (!decodeType) {
        prNum = _num_priors;
    }
    parallel_for(prNum, [&](int p) {
        if (_confidence_threshold != 0) {
            if (_share_location && conf_info_v[p] == -1) {
                return;
            }
            if (!_share_location && !_decrease_label_id && conf_info_h[_num_priors] == 0) {
                return;
            }
        }
        float new_xmin = 0.0f;
        float new_ymin = 0.0f;
        float new_xmax = 0.0f;
        float new_ymax = 0.0f;

        float prior_xmin = prior_data[p*pr_size + 0 + offs];
        float prior_ymin = prior_data[p*pr_size + 1 + offs];
        float prior_xmax = prior_data[p*pr_size + 2 + offs];
        float prior_ymax = prior_data[p*pr_size + 3 + offs];

        float loc_xmin = loc_data[4*p*_num_loc_classes + 0];
        float loc_ymin = loc_data[4*p*_num_loc_classes + 1];
        float loc_xmax = loc_data[4*p*_num_loc_classes + 2];
        float loc_ymax = loc_data[4*p*_num_loc_classes + 3];

        if (!_normalized) {
            prior_xmin /= _image_width;
            prior_ymin /= _image_height;
            prior_xmax /= _image_width;
            prior_ymax /= _image_height;
        }

        if (_code_type == CodeType::CORNER) {
            if (_variance_encoded_in_target) {
                // variance is encoded in target, we simply need to add the offset predictions.
                new_xmin = prior_xmin + loc_xmin;
                new_ymin = prior_ymin + loc_ymin;
                new_xmax = prior_xmax + loc_xmax;
                new_ymax = prior_ymax + loc_ymax;
            } else {
                new_xmin = prior_xmin + variance_data[p*4 + 0] * loc_xmin;
                new_ymin = prior_ymin + variance_data[p*4 + 1] * loc_ymin;
                new_xmax = prior_xmax + variance_data[p*4 + 2] * loc_xmax;
                new_ymax = prior_ymax + variance_data[p*4 + 3] * loc_ymax;
            }
        } else if (_code_type == CodeType::CENTER_SIZE) {
            float prior_width    =  prior_xmax - prior_xmin;
            float prior_height   =  prior_ymax - prior_ymin;
            float prior_center_x = (prior_xmin + prior_xmax) / 2.0f;
            float prior_center_y = (prior_ymin + prior_ymax) / 2.0f;

            float decode_bbox_center_x, decode_bbox_center_y;
            float decode_bbox_width, decode_bbox_height;

            if (_variance_encoded_in_target) {
                // variance is encoded in target, we simply need to restore the offset predictions.
                decode_bbox_center_x = loc_xmin * prior_width  + prior_center_x;
                decode_bbox_center_y = loc_ymin * prior_height + prior_center_y;
                decode_bbox_width  = std::exp(loc_xmax) * prior_width;
                decode_bbox_height = std::exp(loc_ymax) * prior_height;
            } else {
                // variance is encoded in bbox, we need to scale the offset accordingly.
                decode_bbox_center_x = variance_data[p*4 + 0] * loc_xmin * prior_width + prior_center_x;
                decode_bbox_center_y = variance_data[p*4 + 1] * loc_ymin * prior_height + prior_center_y;
                decode_bbox_width    = std::exp(variance_data[p*4 + 2] * loc_xmax) * prior_width;
                decode_bbox_height   = std::exp(variance_data[p*4 + 3] * loc_ymax) * prior_height;
            }

            new_xmin = decode_bbox_center_x - decode_bbox_width  / 2.0f;
            new_ymin = decode_bbox_center_y - decode_bbox_height / 2.0f;
            new_xmax = decode_bbox_center_x + decode_bbox_width  / 2.0f;
            new_ymax = decode_bbox_center_y + decode_bbox_height / 2.0f;
        }

        if (_clip_before_nms) {
            new_xmin = (std::max)(0.0f, (std::min)(1.0f, new_xmin));
            new_ymin = (std::max)(0.0f, (std::min)(1.0f, new_ymin));
            new_xmax = (std::max)(0.0f, (std::min)(1.0f, new_xmax));
            new_ymax = (std::max)(0.0f, (std::min)(1.0f, new_ymax));
        }

        decoded_bboxes[p*4 + 0] = new_xmin;
        decoded_bboxes[p*4 + 1] = new_ymin;
        decoded_bboxes[p*4 + 2] = new_xmax;
        decoded_bboxes[p*4 + 3] = new_ymax;

        decoded_bbox_sizes[p] = (new_xmax - new_xmin) * (new_ymax - new_ymin);
    });
}

void MKLDNNDetectionOutputNode::topk(const int *indicesIn, int *indicesOut, const float *conf, int n, int k) {
    std::partial_sort_copy(indicesIn, indicesIn + n,
                           indicesOut, indicesOut + k,
                           ConfidenceComparator(conf));
}

// bbox decode when needed, and store into box buffer
void MKLDNNDetectionOutputNode::nms_cf(int* indicesIn,
                                    int& detections,
                                    int* indicesOut,
                                    const float* bboxes,
                                    const float* boxSizes) {
    // nms for this class
    int countIn = detections;
    detections = 0;
    for (int i = 0; i < countIn; ++i) {
        const int idx = indicesIn[i];

        bool keep = true;
        for (int k = 0; k < detections; ++k) {
            const int kept_idx = indicesOut[k];
            float overlap = JaccardOverlap(bboxes, boxSizes, idx, kept_idx);
            if (overlap > _nms_threshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            indicesOut[detections] = idx;
            detections++;
        }
    }
}

void MKLDNNDetectionOutputNode::nms_mx(int* indicesIn,
                                    int* detections,
                                    int* indicesOut,
                                    const float* bboxes,
                                    const float* sizes) {
    // Input is candidate for image, output is candidate for each class within image
    int countIn = detections[0];
    detections[0] = 0;

    for (int i = 0; i < countIn; ++i) {
        const int idx = indicesIn[i];
        const int cls = idx/_num_priors;
        const int prior = idx%_num_priors;

        // nms within this class
        int &ndetection = detections[cls];
        int *pindices = indicesOut + cls*_num_priors;

        bool keep = true;
        for (int k = 0; k < ndetection; ++k) {
            const int kept_prior = pindices[k];
            float overlap = 0.0f;
            if (_share_location) {
                overlap = JaccardOverlap(bboxes, sizes, prior, kept_prior);
            } else {
                overlap = JaccardOverlap(bboxes, sizes, cls*_num_priors + prior, cls*_num_priors + kept_prior);
            }
            if (overlap > _nms_threshold) {
                keep = false;
                break;
            }
        }

        if (keep) {
            pindices[ndetection++] = prior;
        }
    }
}

bool MKLDNNDetectionOutputNode::created() const {
    return getType() == DetectionOutput;
}

REG_MKLDNN_PRIM_FOR(MKLDNNDetectionOutputNode, DetectionOutput)
