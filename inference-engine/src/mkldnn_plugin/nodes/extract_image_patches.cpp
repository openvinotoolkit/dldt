// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "list.hpp"
#include "base.hpp"
#include "caseless.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <set>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using details::CaselessEq;

class ExtractImagePatchesImpl : public ExtLayerBase {
public:
    explicit ExtractImagePatchesImpl(const CNNLayer* layer) {
        try {
            std::string errorPrefix = std::string("Layer ") + layer->type + " with name '" + layer->name + "' ";
            if (details::CaselessEq<std::string>()("ExtractImagePatchesLayer", layer->type))
                IE_THROW() << errorPrefix << "is not an instance of ExtractImagePatchesLayer class";

            if (layer->insData.size() != 1 || layer->outData.size() != 1)
                IE_THROW() << errorPrefix << "has incorrect number of input or output edges!"
                    << " Input: " << layer->insData.size() << "; Output: " << layer->outData.size();

            auto inData = layer->insData[0].lock();
            if (inData == nullptr)
                IE_THROW() << errorPrefix << "has nullable input data";

            if (inData->getTensorDesc().getDims().size() != 4)
                IE_THROW() << errorPrefix << "must have 4D input tensor. Actual: " << inData->getTensorDesc().getDims().size();

            if (layer->outData[0]->getTensorDesc().getDims().size() != 4)
                IE_THROW() << errorPrefix << "must have 4D output tensor. Actual: " << layer->outData[0]->getTensorDesc().getDims().size();

            if (inData->getLayout() != NCHW)
                IE_THROW() << errorPrefix << "has unsupported layout: " << inData->getLayout();

            const auto precision = inData->getTensorDesc().getPrecision();
            if (_supported_precisions_sizes.find(precision.size()) == _supported_precisions_sizes.end())
                IE_THROW() << errorPrefix << "has unsupported precision: " << precision.name();

            auto ksizes = layer->GetParamAsUInts("sizes");
            auto strides = layer->GetParamAsUInts("strides");
            auto rates = layer->GetParamAsUInts("rates");
            _auto_pad = layer->GetParamAsString("auto_pad");
            if (!CaselessEq<std::string>()(_auto_pad, "valid")
                    && !CaselessEq<std::string>()(_auto_pad, "same_upper")
                    && !CaselessEq<std::string>()(_auto_pad, "same_lower"))
                IE_THROW() <<  errorPrefix << "has unsupported auto_pad value: " << _auto_pad;
            if (ksizes.size() != 2 || strides.size() != 2 || rates.size() != 2)
                IE_THROW() << errorPrefix << "must have the following attributes with shape {2}: sizes, strides, rates.";

            _ksizes.clear();
            _strides.clear();
            _rates.clear();
            for (size_t i = 0; i < ksizes.size(); i++)
                _ksizes.push_back((int64_t)ksizes[i]);
            for (size_t i = 0; i < strides.size(); i++)
                _strides.push_back((int64_t)strides[i]);
            for (size_t i = 0; i < rates.size(); i++)
                _rates.push_back((int64_t)rates[i]);

            LayerConfig config;

            DataConfig inConfig;
            inConfig.desc = inData->getTensorDesc();
            config.inConfs.push_back(inConfig);

            DataConfig outConfig;
            outConfig.desc = layer->outData[0]->getTensorDesc();
            outConfig.desc.setPrecision(inConfig.desc.getPrecision());
            outConfig.desc.setLayout(inConfig.desc.getLayout());
            config.outConfs.push_back(outConfig);

            config.dynBatchSupport = false;
            confs.push_back(config);
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        switch (inputs[0]->getTensorDesc().getPrecision().size()) {
            case 1: {
                process_data<PrecisionTrait<Precision::U8>::value_type>(inputs, outputs);
                break;
            }
            case 2: {
                process_data<PrecisionTrait<Precision::U16>::value_type>(inputs, outputs);
                break;
            }
            case 4: {
                process_data<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs);
                break;
            }
            case 8: {
                process_data<PrecisionTrait<Precision::U64>::value_type>(inputs, outputs);
                break;
            }
            default: {
                if (resp) {
                    std::string errorMsg = "ExtractImagePatches layer does not support precision '"
                            + std::string(inputs[0]->getTensorDesc().getPrecision().name()) + "'";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return GENERAL_ERROR;
            }
        }

        return OK;
    }

    template<typename T>
    void process_data(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs) noexcept {
        const T* src_data = inputs[0]->cbuffer().as<const T*>() +
            inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        T* dst_data = outputs[0]->buffer().as<T*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const auto& inDims = inputs[0]->getTensorDesc().getDims();
        const size_t inDimsSize = inDims.size(); // Must always be 4 according to the specs.

        const size_t BATCH = 0, CHANNEL = 1, HIGHT = 0, WIDTH = 1;

        const int64_t IC = inDims[CHANNEL];
        const int64_t IH = inDims[inDimsSize - 2];
        const int64_t IW = inDims[inDimsSize - 1];

        const auto& outDims = outputs[0]->getTensorDesc().getDims();
        const size_t outDimsSize = outDims.size(); // Must always be 4 according to the specs.

        const int64_t OB = outDims[BATCH];
        //const int64_t OC = outDims[CHANNEL]; // Must always be KH * KW * IC according to the specs.
        const int64_t OH = outDims[outDimsSize - 2];
        const int64_t OW = outDims[outDimsSize - 1];

        const int64_t KH = _ksizes[HIGHT];
        const int64_t KW = _ksizes[WIDTH];
        const int64_t SH = _strides[HIGHT];
        const int64_t SW = _strides[WIDTH];
        const int64_t RH = _rates[HIGHT];
        const int64_t RW = _rates[WIDTH];

        int64_t iwStep = KW + (RW - 1) * (KW - 1);
        int64_t ihStep = KH + (RH - 1) * (KH - 1);

        int64_t PL = 0, PT = 0;
        if (!CaselessEq<std::string>()(_auto_pad, "valid")) {
            int64_t PW = (std::ceil(1.f * IW/SW) - 1) * SW + iwStep - IW;
            int64_t PH = (std::ceil(1.f * IH/SH) - 1) * SH + ihStep - IH;

            if ((PW > 0) && (PW < iwStep)) {
                if (PW % 2 == 1) {
                    if (CaselessEq<std::string>()(_auto_pad, "same_lower")) {
                        PL = (PW + 1) / 2;
                    } else if (CaselessEq<std::string>()(_auto_pad, "same_upper")) {
                        PL = (PW - 1) / 2;
                    }
                } else {
                    PL = PW / 2;
                }
            }
            if ((PH > 0) && (PH < ihStep)) {
                if (PH % 2 == 1) {
                    if (CaselessEq<std::string>()(_auto_pad, "same_lower")) {
                        PT = (PH + 1) / 2;
                    } else if (CaselessEq<std::string>()(_auto_pad, "same_upper")) {
                        PT = (PH - 1) / 2;
                    }
                } else {
                    PT = PH / 2;
                }
            }
        }
        const std::vector<int64_t> ostrides = {KH * KW * IC * OH * OW, KW * IC * OH * OW, IC * OH * OW, OH * OW};
        const std::vector<int64_t> istrides = {IC * IH * IW, IH * IW, IW};
        auto thread_body = [&](const int64_t ob, const int64_t kh, const int64_t kw, const int64_t ic) {
            const int64_t iw_start = kw * RW - PL;
            const int64_t iw_stop = iw_start + OW * SW;
            const int64_t ih_start = kh * RH - PT;
            const int64_t ih_stop = ih_start + OH * SH;
            int64_t dst_idx = ob * ostrides[0]  + kh * ostrides[1] + kw * ostrides[2] + ic * ostrides[3];
            int64_t ishift = ob * istrides[0] + ic * istrides[1] + ih_start * istrides[2];
            for (int64_t ih = ih_start; ih < ih_stop; ih += SH, ishift += SH * IW) {
                for (int64_t iw = iw_start; iw < iw_stop; iw += SW, dst_idx++) {
                    if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) {
                        dst_data[dst_idx] = T(0);
                    } else {
                        dst_data[dst_idx] = src_data[ishift + iw];
                    }
                }
            }
        };
        parallel_for4d(OB, KH, KW, IC, thread_body);
    }

private:
    std::vector<int64_t> _ksizes;
    std::vector<int64_t> _strides;
    std::vector<int64_t> _rates;
    std::string _auto_pad;

    static const std::set<size_t> _supported_precisions_sizes;
};

const std::set<size_t> ExtractImagePatchesImpl::_supported_precisions_sizes = {1, 2, 4, 8};

REG_FACTORY_FOR(ExtractImagePatchesImpl, ExtractImagePatches);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
