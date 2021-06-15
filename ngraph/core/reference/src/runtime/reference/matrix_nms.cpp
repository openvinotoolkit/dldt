// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/matrix_nms.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <vector>
#include "ngraph/runtime/reference/matrix_nms.hpp"
#include "ngraph/shape.hpp"

using namespace ngraph;
using namespace ngraph::runtime::reference;

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace
            {
                template <typename T, bool gaussian>
                struct decay_score;

                template <typename T>
                struct decay_score<T, true>
                {
                    T operator()(T iou, T max_iou, T sigma)
                    {
                        return std::exp((max_iou * max_iou - iou * iou) * sigma);
                    }
                };

                template <typename T>
                struct decay_score<T, false>
                {
                    T operator()(T iou, T max_iou, T sigma) { return (1. - iou) / (1. - max_iou); }
                };

                template <class T>
                static inline T BBoxArea(const T* box, const bool normalized)
                {
                    if (box[2] < box[0] || box[3] < box[1])
                    {
                        // If coordinate values are is invalid
                        // (e.g. xmax < xmin or ymax < ymin), return 0.
                        return static_cast<T>(0.);
                    }
                    else
                    {
                        const T w = box[2] - box[0];
                        const T h = box[3] - box[1];
                        if (normalized)
                        {
                            return w * h;
                        }
                        else
                        {
                            // If coordinate values are not within range [0, 1].
                            return (w + 1) * (h + 1);
                        }
                    }
                }

                template <class T>
                static inline T
                    intersectionOverUnion(const T* box1, const T* box2, const bool normalized)
                {
                    if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
                        box2[3] < box1[1])
                    {
                        return static_cast<T>(0.);
                    }
                    else
                    {
                        const T inter_xmin = std::max(box1[0], box2[0]);
                        const T inter_ymin = std::max(box1[1], box2[1]);
                        const T inter_xmax = std::min(box1[2], box2[2]);
                        const T inter_ymax = std::min(box1[3], box2[3]);
                        T norm = normalized ? static_cast<T>(0.) : static_cast<T>(1.);
                        T inter_w = inter_xmax - inter_xmin + norm;
                        T inter_h = inter_ymax - inter_ymin + norm;
                        const T inter_area = inter_w * inter_h;
                        const T bbox1_area = BBoxArea<T>(box1, normalized);
                        const T bbox2_area = BBoxArea<T>(box2, normalized);
                        return inter_area / (bbox1_area + bbox2_area - inter_area);
                    }
                }

            } // namespace

            template <typename T, bool gaussian>
            void nms_matrix(const T* boxes_data,
                            const Shape& boxes_data_shape,
                            const T* scores_data,
                            const Shape& scores_data_shape,
                            const T score_threshold,
                            const T post_threshold,
                            const float sigma,
                            const int64_t top_k,
                            const bool normalized,
                            std::vector<int>* selected_indices,
                            std::vector<T>* decayed_scores)
            {
                int64_t boxes_num = static_cast<int64_t>(boxes_data_shape[1]);
                int64_t box_size = static_cast<int64_t>(boxes_data_shape[2]);

                std::vector<int32_t> candidate_index(boxes_num);
                std::iota(candidate_index.begin(), candidate_index.end(), 0);
                auto end = std::remove_if(candidate_index.begin(),
                                          candidate_index.end(),
                                          [&scores_data, score_threshold](int32_t idx) {
                                              return scores_data[idx] <= score_threshold;
                                          });

                int64_t original_size = std::distance(candidate_index.begin(), end);
                if (original_size <= 0)
                {
                    return;
                }
                if (top_k > -1 && original_size > top_k)
                {
                    original_size = top_k;
                }

                std::partial_sort(candidate_index.begin(),
                                  candidate_index.begin() + original_size,
                                  end,
                                  [&scores_data](int32_t a, int32_t b) {
                                      return scores_data[a] > scores_data[b];
                                  });

                std::vector<T> iou_matrix((original_size * (original_size - 1)) >> 1);
                std::vector<T> iou_max(original_size);

                iou_max[0] = 0.;
                for (int64_t i = 1; i < original_size; i++)
                {
                    T max_iou = 0.;
                    auto idx_a = candidate_index[i];
                    for (int64_t j = 0; j < i; j++)
                    {
                        auto idx_b = candidate_index[j];
                        auto iou = intersectionOverUnion<T>(boxes_data + idx_a * box_size,
                                                            boxes_data + idx_b * box_size,
                                                            normalized);
                        max_iou = std::max(max_iou, iou);
                        iou_matrix[i * (i - 1) / 2 + j] = iou;
                    }
                    iou_max[i] = max_iou;
                }

                if (scores_data[candidate_index[0]] > post_threshold)
                {
                    selected_indices->push_back(candidate_index[0]);
                    decayed_scores->push_back(scores_data[candidate_index[0]]);
                }

                decay_score<T, gaussian> decay_fn;
                for (int64_t i = 1; i < original_size; i++)
                {
                    T min_decay = 1.;
                    for (int64_t j = 0; j < i; j++)
                    {
                        auto max_iou = iou_max[j];
                        auto iou = iou_matrix[i * (i - 1) / 2 + j];
                        auto decay = decay_fn(iou, max_iou, sigma);
                        min_decay = std::min(min_decay, decay);
                    }
                    auto ds = min_decay * scores_data[candidate_index[i]];
                    if (ds <= post_threshold)
                        continue;
                    selected_indices->push_back(candidate_index[i]);
                    decayed_scores->push_back(ds);
                }
            }

            void matrix_nms(const float* boxes_data,
                            const Shape& boxes_data_shape,
                            const float* scores_data,
                            const Shape& scores_data_shape,
                            op::util::NmsBase::SortResultType sort_result_type,
                            bool sort_result_across_batch,
                            float score_threshold,
                            int nms_top_k,
                            int keep_top_k,
                            int background_class,
                            const op::v8::MatrixNms::DecayFunction decay_function,
                            float gaussian_sigma,
                            float post_threshold,
                            float* selected_outputs,
                            const Shape& selected_outputs_shape,
                            int64_t* selected_indices,
                            const Shape& selected_indices_shape,
                            int64_t* valid_outputs)
            {
                // boxes shape: {num_batches, num_boxes, 4}
                // scores shape: {num_batches, num_classes, num_boxes}
                int64_t num_batches = static_cast<int64_t>(scores_data_shape[0]);
                int64_t num_classes = static_cast<int64_t>(scores_data_shape[1]);
                int64_t num_boxes = static_cast<int64_t>(boxes_data_shape[1]);
                int64_t box_shape = static_cast<int64_t>(boxes_data_shape[2]);
                std::vector<float> detections;
                std::vector<int> indices;
                std::vector<int> num_per_batch;
                detections.reserve(6 * num_batches * num_classes * num_boxes);

                int64_t background_label = 0;

                bool normalized = true;
                for (int64_t batch = 0; batch < num_batches; batch++)
                {
                    const float* boxesPtr = boxes_data + batch * num_boxes * 4;
                    std::vector<int> all_indices;
                    std::vector<float> all_scores;
                    std::vector<float> all_classes;
                    size_t num_det = 0;

                    for (int64_t class_idx = 0; class_idx < num_classes; class_idx++)
                    {
                        if (class_idx == background_label)
                            continue;
                        const float* scoresPtr =
                            scores_data + batch * (num_classes * num_boxes) + class_idx * num_boxes;
                        if (decay_function == op::v8::MatrixNms::DecayFunction::GAUSSIAN)
                        {
                            nms_matrix<float, true>(boxesPtr,
                                                    boxes_data_shape,
                                                    scoresPtr,
                                                    scores_data_shape,
                                                    score_threshold,
                                                    post_threshold,
                                                    gaussian_sigma,
                                                    nms_top_k,
                                                    normalized,
                                                    &all_indices,
                                                    &all_scores);
                        }
                        else
                        {
                            nms_matrix<float, false>(boxesPtr,
                                                     boxes_data_shape,
                                                     scoresPtr,
                                                     scores_data_shape,
                                                     score_threshold,
                                                     post_threshold,
                                                     gaussian_sigma,
                                                     nms_top_k,
                                                     normalized,
                                                     &all_indices,
                                                     &all_scores);
                        }
                        for (size_t i = 0; i < all_indices.size() - num_det; i++)
                        {
                            all_classes.push_back(static_cast<float>(class_idx));
                        }
                        num_det = all_indices.size();
                    }

                    if (num_det <= 0)
                    {
                        break;
                    }

                    if (keep_top_k > -1)
                    {
                        auto k = static_cast<size_t>(keep_top_k);
                        if (num_det > k)
                            num_det = k;
                    }

                    std::vector<int32_t> perm(all_indices.size());
                    std::iota(perm.begin(), perm.end(), 0);

                    std::partial_sort(perm.begin(),
                                      perm.begin() + num_det,
                                      perm.end(),
                                      [&all_scores](int lhs, int rhs) {
                                          return all_scores[lhs] > all_scores[rhs];
                                      });

                    for (size_t i = 0; i < num_det; i++)
                    {
                        auto p = perm[i];
                        auto idx = all_indices[p];
                        auto cls = all_classes[p];
                        auto score = all_scores[p];
                        auto bbox = boxesPtr + idx * box_shape;

                        indices.push_back(batch * num_boxes + idx);
                        detections.push_back(cls);
                        detections.push_back(score);
                        for (int j = 0; j < box_shape; j++)
                        {
                            detections.push_back(bbox[j]);
                        }
                    }
                    num_per_batch.push_back(num_det);
                }

                std::copy(indices.begin(), indices.end(), selected_indices);
                std::copy(num_per_batch.begin(), num_per_batch.end(), valid_outputs);
                std::copy(detections.begin(), detections.end(), selected_outputs);
            }

            void matrix_nms_postprocessing(const HostTensorVector& outputs,
                                           const ngraph::element::Type output_type,
                                           const std::vector<float>& selected_outputs,
                                           const std::vector<int64_t>& selected_indices,
                                           const std::vector<int64_t>& valid_outputs)
            {
                int64_t total_num = std::accumulate(valid_outputs.begin(), valid_outputs.end(), 0);
                outputs[0]->set_shape(Shape{static_cast<size_t>(total_num), 6});
                float* ptr = outputs[0]->get_data_ptr<float>();
                memcpy(ptr, selected_outputs.data(), total_num * sizeof(float) * 6);

                if (outputs.size() >= 2)
                {
                    outputs[1]->set_shape(Shape{static_cast<size_t>(total_num), 1});
                    if (output_type == ngraph::element::i64)
                    {
                        int64_t* indices_ptr = outputs[1]->get_data_ptr<int64_t>();
                        memcpy(indices_ptr, selected_indices.data(), total_num * sizeof(int64_t));
                    }
                    else
                    {
                        int32_t* indices_ptr = outputs[1]->get_data_ptr<int32_t>();
                        for (size_t i = 0; i < (size_t)total_num; ++i)
                        {
                            indices_ptr[i] = static_cast<int32_t>(selected_indices[i]);
                        }
                    }
                }

                if (outputs.size() >= 3)
                {
                    if (output_type == ngraph::element::i64)
                    {
                        int64_t* valid_outputs_ptr = outputs[2]->get_data_ptr<int64_t>();
                        std::copy(valid_outputs.begin(), valid_outputs.end(), valid_outputs_ptr);
                    }
                    else
                    {
                        int32_t* valid_outputs_ptr = outputs[2]->get_data_ptr<int32_t>();
                        for (size_t i = 0; i < (size_t)valid_outputs.size(); ++i)
                        {
                            valid_outputs_ptr[i] = static_cast<int32_t>(valid_outputs[i]);
                        }
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
