// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <memory>

namespace vpu {

namespace {

const int numDeltasWeights = 4;

VPU_PACKED(ExpDetectionOutputParams {
    float   deltas_weights[numDeltasWeights];
    float   max_delta_log_wh;
    float   nms_threshold;
    float   score_threshold;
    int32_t max_detections_per_image;
    int32_t num_classes;
    int32_t post_nms_count;
    int32_t class_agnostic_box_regression;
};)

class ExpDetectionOutputStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ExpDetectionOutputStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        for (const auto& inEdge : inputEdges()) {
            stridesInfo.setInput(inEdge, StridesRequirement::compact());
        }
        for (const auto& outEdge : outputEdges()) {
            stridesInfo.setOutput(outEdge, StridesRequirement::compact());
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this,
             {{DataType::FP16}, {DataType::FP16}, {DataType::FP16}, {DataType::FP16}},
             {{DataType::FP16}, {DataType::S32}, {DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto& params = attrs().get<ExpDetectionOutputParams>("params");

        serializer.append(params);
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto inputBoxes = inputEdges()[0]->input();
        auto inputDeltas = inputEdges()[1]->input();
        auto inputScores = inputEdges()[2]->input();
        auto inputIMinfo = inputEdges()[3]->input();
        auto outputBoxes = outputEdges()[0]->output();
        auto outputClasses = outputEdges()[1]->output();
        auto outputScores = outputEdges()[2]->output();

        inputBoxes->serializeBuffer(serializer);
        inputDeltas->serializeBuffer(serializer);
        inputScores->serializeBuffer(serializer);
        inputIMinfo->serializeBuffer(serializer);
        outputBoxes->serializeBuffer(serializer);
        outputClasses->serializeBuffer(serializer);
        outputScores->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseExpDetectionOutput(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    auto expDetectionOutput = ngraph::as_type_ptr<ngraph::opset6::ExperimentalDetectronDetectionOutput>(node);
    VPU_THROW_UNLESS(expDetectionOutput != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    IE_ASSERT(inputs.size() == 4);
    IE_ASSERT(outputs.size() == 3);

    ExpDetectionOutputParams params;
    const auto attrs = expDetectionOutput->get_attrs();
    const auto deltas_weights = attrs.deltas_weights;
    IE_ASSERT(deltas_weights.size() == numDeltasWeights);
    for (int i = 0; i < numDeltasWeights; ++i)
        params.deltas_weights[i] = deltas_weights[i];

    params.max_delta_log_wh = attrs.max_delta_log_wh;
    params.nms_threshold = attrs.nms_threshold;
    params.score_threshold = attrs.score_threshold;
    params.max_detections_per_image = attrs.max_detections_per_image;
    params.num_classes = attrs.num_classes;
    params.post_nms_count = attrs.post_nms_count;
    params.class_agnostic_box_regression = attrs.class_agnostic_box_regression;

    auto inputBoxes    = inputs[0];   // [numRois][4]
    auto inputDeltas   = inputs[1];   // [numRois]([numClasses][4])
    auto inputScores   = inputs[2];   // [numRois][numClasses]
    auto inputIMinfo   = inputs[3];   // [2]
    auto outputBoxes   = outputs[0];  // [maxDetections][4]
    auto outputClasses = outputs[1];  // [maxDetections]
    auto outputScores  = outputs[2];  // [maxDetections]

    // from layer point of view, they are not N or C at all; but layout/order require:
    //  2-dim => NC [N][C] [input Boxes, Deltas, Scores, IMinfo; output Boxes]
    //  1-dim => C  [C]    [output Classes, Scores]

    const int numRois       = inputBoxes->desc().dim(Dim::N);
    const int numClasses    = inputScores->desc().dim(Dim::C);
    const int maxDetections = params.max_detections_per_image;

    IE_ASSERT((inputBoxes->desc().dims().size() == 2) &&
              (inputBoxes->desc().dim(Dim::C) == 4));
    IE_ASSERT((inputDeltas->desc().dims().size() == 2) &&
              (inputDeltas->desc().dim(Dim::N) == numRois) &&
              (inputDeltas->desc().dim(Dim::C) == numClasses * 4));
    IE_ASSERT((inputScores->desc().dims().size() == 2) &&
              (inputScores->desc().dim(Dim::N) == numRois));
    IE_ASSERT((inputIMinfo->desc().dims().size() == 2) &&
              (inputIMinfo->desc().dim(Dim::N) == 1) &&
              (inputIMinfo->desc().dim(Dim::C) >= 2));

    IE_ASSERT((outputBoxes->desc().dims().size() == 2) &&
              (outputBoxes->desc().dim(Dim::N) >= maxDetections) &&
              (outputBoxes->desc().dim(Dim::C) == 4));
    IE_ASSERT((outputClasses->desc().dims().size() == 1) &&
              (outputClasses->desc().dim(Dim::C) >= maxDetections));
    IE_ASSERT((outputScores->desc().dims().size() == 1) &&
              (outputScores->desc().dim(Dim::C) >= maxDetections));

    auto stage = model->addNewStage<ExpDetectionOutputStage>(expDetectionOutput->get_friendly_name(), StageType::ExpDetectionOutput, expDetectionOutput, inputs, outputs);

    stage->attrs().set("params", params);
}

}  // namespace vpu
