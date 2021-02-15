// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <string>
#include <vector>
#include <list>
#include <set>
#include <unordered_set>
#include <memory>

namespace vpu {

namespace {

class PadStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<PadStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        orderInfo.setOutput(outputEdge(0), input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        // TODO: try merge with last dimension
        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::CanBeLimited;
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();

        auto perm = input->desc().dimsOrder().toPermutation();
        IE_ASSERT(perm.size() <= 4);

        auto pad_value = attrs().get<float>("pad_value");
        auto pad_mode = attrs().get<PadMode>("pad_mode");
        const auto& pads_begin = attrs().get<DimValues>("pads_begin");
        const auto& pads_end = attrs().get<DimValues>("pads_end");

        int i = 0;
        for (; i < perm.size(); ++i) {
            serializer.append(static_cast<uint32_t>(pads_begin.get(perm[i], 0)));
            serializer.append(static_cast<uint32_t>(pads_end.get(perm[i], 0)));
        }
        for (; i < 4; ++i) {
            serializer.append(static_cast<uint32_t>(0));
            serializer.append(static_cast<uint32_t>(0));
        }

        serializer.append(static_cast<float>(pad_value));
        serializer.append(static_cast<uint32_t>(pad_mode));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parsePad(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::PadLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    const auto ndims = inputs[0]->desc().dimsOrder().numDims();
    VPU_THROW_UNLESS(ndims == 3 || ndims == 4, "Layer %s support only 3D and 4D input", layer->name);

    IE_ASSERT(layer->pads_begin.size() <= 4);
    IE_ASSERT(layer->pads_end.size() <= 4);

    unsigned int pads_beginN = 0;
    unsigned int pads_beginC = 0;
    unsigned int pads_beginH = 0;
    unsigned int pads_beginW = 0;

    unsigned int pads_endN = 0;
    unsigned int pads_endC = 0;
    unsigned int pads_endH = 0;
    unsigned int pads_endW = 0;

    switch (ndims) {
        case 4:
            pads_beginN = layer->pads_begin[0];
            pads_beginC = layer->pads_begin[1];
            pads_beginH = layer->pads_begin[2];
            pads_beginW = layer->pads_begin[3];

            pads_endN = layer->pads_end[0];
            pads_endC = layer->pads_end[1];
            pads_endH = layer->pads_end[2];
            pads_endW = layer->pads_end[3];
            break;
        case 3:
            pads_beginN = 0;
            pads_beginC = layer->pads_begin[0];
            pads_beginH = layer->pads_begin[1];
            pads_beginW = layer->pads_begin[2];

            pads_endN = 0;
            pads_endC = layer->pads_end[0];
            pads_endH = layer->pads_end[1];
            pads_endW = layer->pads_end[2];
            break;
    }

    DimValues pads_begin;
    pads_begin.set(Dim::W, pads_beginW);
    pads_begin.set(Dim::H, pads_beginH);
    pads_begin.set(Dim::C, pads_beginC);
    pads_begin.set(Dim::N, pads_beginN);

    DimValues pads_end;
    pads_end.set(Dim::W, pads_endW);
    pads_end.set(Dim::H, pads_endH);
    pads_end.set(Dim::C, pads_endC);
    pads_end.set(Dim::N, pads_endN);

    _stageBuilder->addPadStage(
        model,
        layer->name,
        layer,
        static_cast<PadMode>(layer->pad_mode),
        layer->pad_value,
        pads_begin,
        pads_end,
        inputs[0],
        outputs[0]);
}

Stage StageBuilder::addPadStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        PadMode padMode,
        float pad_value,
        const DimValues& pads_begin,
        const DimValues& pads_end,
        const Data& input,
        const Data& output) {
    auto stage = model->addNewStage<PadStage>(
        name,
        StageType::Pad,
        layer,
        {input},
        {output});

    stage->attrs().set<float>("pad_value", pad_value);
    stage->attrs().set<PadMode>("pad_mode", padMode);
    stage->attrs().set<DimValues>("pads_begin", pads_begin);
    stage->attrs().set<DimValues>("pads_end", pads_end);

    return stage;
}

}  // namespace vpu
