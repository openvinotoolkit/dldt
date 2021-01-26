// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_concat_node.h"

#include <map>
#include <utility>
#include <vector>
#include <mkldnn_extension_utils.h>

#include "details/ie_exception.hpp"
#include <legacy/ie_layers.h>
#include "mkldnn.hpp"
#include "mkldnn/iml_type_mapper.h"
#include "mkldnn_dims.h"
#include "mkldnn_edge.h"
#include "mkldnn_memory.h"
#include "ie_parallel.hpp"
#include "mkldnn_conv_node.h"
#include "mkldnn_quantize_node.h"
#include "mkldnn_pooling_node.h"
#include "mkldnn_eltwise_node.h"
#include <limits>
#include "common/cpu_memcpy.h"
#include "common/tensor_desc_creator.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

namespace {
    constexpr size_t channelAxis = 1lu;
}

MKLDNNConcatNode::MKLDNNConcatNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNConcatNode::getSupportedDescriptors() {
    auto * conLayer = dynamic_cast<ConcatLayer*>(getCnnLayer().get());

    if (conLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert concat layer.";

    axis = conLayer->_axis;

    if (getParentEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();
    auto& firstParentDims = getParentEdgeAt(0)->getDims();
    for (size_t i = 1; i < getParentEdges().size(); i++) {
        auto& dims = getParentEdgeAt(i)->getDims();
        bool incorrectDims = false;
        for (size_t j = 0; j < firstParentDims.ndims(); j++) {
            if (j == axis)
                continue;
            if (dims.ndims() != firstParentDims.ndims() || firstParentDims[j] != dims[j]) {
                incorrectDims = true;
                break;
            }
        }
        if (incorrectDims || firstParentDims.ndims() == 0) {
            THROW_IE_EXCEPTION << "Incorrect input dimensions for concat node " << getName();
        }
    }
}

void MKLDNNConcatNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    inputPrecision = getCnnLayer()->insData[0].lock()->getPrecision();
    bool isMixedPrecision = false;
    for (int i = 1; i < getCnnLayer()->insData.size(); i++) {
        if (getCnnLayer()->insData[0].lock()->getPrecision() != getCnnLayer()->insData[i].lock()->getPrecision()) {
            isMixedPrecision = true;
            break;
        }
    }

    // Concat doesn't support different precision on inputs so fallback on FP32 in such case
    if (isMixedPrecision)
        inputPrecision = Precision::FP32;

    // Concat supports only equal precisions for inputs and output
    outputPrecision = inputPrecision;

    auto& dstDims = getChildEdgeAt(0)->getDims();
    std::vector<TensorDescCreatorTypes> tdCreatorTypes = {TensorDescCreatorTypes::ncsp, TensorDescCreatorTypes::nspc};

    // check if blocked layouts are available
    if (dstDims.ndims() > channelAxis) {
        for (auto item : { std::make_pair(8lu, TensorDescCreatorTypes::nCsp8c), std::make_pair(16lu, TensorDescCreatorTypes::nCsp16c)}) {
            SizeVector blkDims = dstDims.ToSizeVector();
            if (blkDims[channelAxis] % item.first)
                continue;

            bool blocked = true;
            for (size_t i = 0; i < getParentEdges().size(); i++) {
                auto& srcDims = getParentEdgeAt(i)->getDims();
                if (srcDims[channelAxis] % item.first) {
                    blocked = false;
                    break;
                }
            }
            if (blocked) {
                tdCreatorTypes.push_back(item.second);
            }
        }
    }

    std::vector<size_t> pdIndexesToReuse;

    auto& creatorsMap = TensorDescCreator::getCommonCreators();
    auto itrRange = TensorDescCreator::makeFilteredRange(creatorsMap, static_cast<unsigned>(dstDims.ndims()), tdCreatorTypes);
    for (auto itr = itrRange.first; itr != itrRange.second; ++itr) {
        InferenceEngine::LayerConfig config;

        config.dynBatchSupport = true;
        config.outConfs.resize(1);
        config.outConfs[0].inPlace = -1;
        config.outConfs[0].constant = false;
        config.outConfs[0].desc = itr->second->createDesc(outputPrecision, dstDims.ToSizeVector());
        memory::format_tag outFmt = MKLDNNMemoryDesc(config.outConfs[0].desc).getFormat();

        config.inConfs.resize(getParentEdges().size());

        for (size_t i = 0; i < getParentEdges().size(); ++i) {
            config.inConfs[i].inPlace = -1;
            config.inConfs[i].constant = false;
            config.inConfs[i].desc = MKLDNNExtensionUtils::getUninitTensorDesc(
                    itr->second->createDesc(inputPrecision, getParentEdgeAt(i)->getDims().ToSizeVector()));
        }
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref, outFmt);
        if (itr->first != TensorDescCreatorTypes::nspc) {
            pdIndexesToReuse.push_back(supportedPrimitiveDescriptors.size() - 1);
        }
    }

    if (axis != channelAxis)
        return;

    // Optimized inplace case

    for (auto refPdIndex : pdIndexesToReuse) {
        const auto& refConfig = supportedPrimitiveDescriptors[refPdIndex].getConfig();
        auto config = refConfig;

        const auto& order = refConfig.outConfs[0].desc.getBlockingDesc().getOrder();
        const auto& blkDims = refConfig.outConfs[0].desc.getBlockingDesc().getBlockDims();
        auto numOfDim = blkDims.size();

        SizeVector offsets(numOfDim, 0lu);
        SizeVector strides(numOfDim);
        strides.back() = 1lu;
        size_t offset = (std::numeric_limits<size_t>::max)();

        for (size_t i = 2; i <= numOfDim; i++) {
            if (numOfDim - i < axis) {
                strides[numOfDim - i] = (std::numeric_limits<size_t>::max)();
            } else {
                strides[numOfDim - i] = strides[numOfDim - i + 1] * blkDims[numOfDim - i + 1];
            }
        }

        config.outConfs[0].desc = TensorDesc(outputPrecision, dstDims.ToSizeVector(), {blkDims, order, offset, offsets, strides});
        memory::format_tag outFmt = MKLDNNMemoryDesc(config.outConfs[0].desc).getFormat();

        for (size_t i = 0; i < getParentEdges().size(); i++) {
            const auto& srcBlkDims = refConfig.inConfs[i].desc.getBlockingDesc().getBlockDims();
            const auto& dims = refConfig.inConfs[i].desc.getDims();

            config.inConfs[i].inPlace = 0;
            config.inConfs[i].desc = TensorDesc(inputPrecision, dims, {srcBlkDims, order, offset, offsets, strides});
        }
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, outFmt);
    }
}

void MKLDNNConcatNode::selectOptimalPrimitiveDescriptor() {
    std::vector<size_t> canSelectPrimitive;

    bool canOptimize = true;

    // The double connection marks that some tensor should
    // be replicated. Inplace approach is not applicable
    // for that case.
    for (int i = 0; i < getParentEdges().size(); i++) {
        for (int j = i + 1; j < getParentEdges().size(); j++) {
            if (getParentEdgeAt(i) == getParentEdgeAt(j)) canOptimize = false;
        }
    }

    if (axis != channelAxis) {
        canOptimize = false;
    }

    std::map<PartialBlkDesc, size_t> formatFrequency;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        auto parent = parentEdge->getParent();

        auto parent_pdesc = parent->getSelectedPrimitiveDescriptor();
        if (parent_pdesc == nullptr)
            continue;

        const auto &parent_config = parent_pdesc->getConfig();
        int outputIndex = parentEdge->getInputNum();
        if (outputIndex < 0 || outputIndex >= parent_config.outConfs.size())
            THROW_IE_EXCEPTION << "Cannot find index of output node";
        const auto &port_desc = parent_config.outConfs[outputIndex].desc;
        if (port_desc.getLayout() == Layout::ANY)
            continue;
        auto partial_format_desc = PartialBlkDesc::extractFrom(port_desc);
        formatFrequency[partial_format_desc] += 1;
    }
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto childEdge = getChildEdgeAt(i);
        auto child = childEdge->getChild();
        const auto *prim_desc = child->getSelectedPrimitiveDescriptor();
        if (prim_desc == nullptr)
            continue;

        const auto &config = prim_desc->getConfig();
        int inputIndex = childEdge->getOutputNum();
        if (inputIndex < 0 || inputIndex >= config.inConfs.size())
            THROW_IE_EXCEPTION << "Cannot find index of output node";
        const auto &port_desc = config.inConfs[inputIndex].desc;
        if (port_desc.getLayout() == Layout::ANY)
            continue;
        auto partial_format_desc = PartialBlkDesc::extractFrom(port_desc);
        formatFrequency[partial_format_desc] += 1;
    }

    size_t maxCount = 0;
    auto convertTo = PartialBlkDesc::makePlain(getChildEdgeAt(0)->getDims().ToSizeVector());
    for (auto &it : formatFrequency) {
        if (it.second > maxCount) {
            maxCount = it.second;
            convertTo = it.first;
        }
    }

    if (convertTo.isAutoExtendedWith(getChildEdgeAt(0)->getDims().ToSizeVector()))
        convertTo = PartialBlkDesc::makePlain(getChildEdgeAt(0)->getDims().ToSizeVector());
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        if (convertTo.isAutoExtendedWith(getParentEdgeAt(i)->getDims().ToSizeVector()))
            convertTo = PartialBlkDesc::makePlain(getChildEdgeAt(0)->getDims().ToSizeVector());
    }

    for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); ++i) {
        if (PartialBlkDesc::extractFrom(supportedPrimitiveDescriptors[i].getConfig().outConfs[0].desc) == convertTo) {
            if (IMPLICATION(supportedPrimitiveDescriptors[i].getImplementationType() == impl_desc_type::unknown, canOptimize)) {
                canSelectPrimitive.push_back(i);
            }
        }
    }

    if (canSelectPrimitive.size() == 1) {
        selectPrimitiveDescriptorByIndex(static_cast<int>(canSelectPrimitive[0]));
        return;
    }

    // if there are more then one PD with similar data layouts - select the optimized one
    for (auto indx : canSelectPrimitive) {
        if (supportedPrimitiveDescriptors[indx].getImplementationType() == impl_desc_type::unknown) {
            selectPrimitiveDescriptorByIndex(static_cast<int>(indx));
            return;
        }
    }

    // if there are no matching data layouts, select first optimized implementation
    for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); i++) {
        if (canOptimize && supportedPrimitiveDescriptors[i].getImplementationType() == impl_desc_type::unknown) {
            selectPrimitiveDescriptorByIndex(static_cast<int>(i));
            return;
        }
    }

    selectPrimitiveDescriptorByIndex(0);
}

bool MKLDNNConcatNode::created() const {
    return getType() == Concatenation;
}

bool MKLDNNConcatNode::isOptimized() const {
    return getSelectedPrimitiveDescriptor() && getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].inPlace >= 0;
}

void MKLDNNConcatNode::createPrimitive() {
    if (prim || isOptimized())
        return;

    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    //check if selected Tensor descriptor has nspc layout and concat axis is C
    if (axis == channelAxis && getChildEdgeAt(0)->getMemory().GetDesc().isTailCFormat()) {
        canOptimizeNspc = true;
        return;
    }

    std::vector<memory::desc> srcs_d;

    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto& srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr()) {
            auto parent = getParentEdgeAt(i)->getParent();
            THROW_IE_EXCEPTION << "Source memory from " << parent->getName() << " didn't allocate for node "
                               << getName() << ".";
        }

        auto desc = srcMemPtr->GetDescriptor();
        auto dims = getParentEdgeAt(i)->getDims();
        for (size_t j = 0; j < dims.ndims(); j++) {
            desc.data.dims[j] = dims[j];
        }

        srcs_d.emplace_back(desc);
    }

    auto desc = getChildEdgeAt(0)->getMemory().GetDescriptor();
    auto dims = getChildEdgeAt(0)->getDims();
    for (size_t i = 0; i < dims.ndims(); i++) {
        desc.data.dims[i] = dims[i];
        desc.data.padded_dims[i] = dims[i];
    }

    auto primitive_desc = concat::primitive_desc(desc, static_cast<int>(axis), srcs_d, getEngine());
    prim.reset(new concat(primitive_desc));
}

size_t MKLDNNConcatNode::inverseOrder(const SizeVector& order, size_t axis) {
    for (size_t i = 0; i < order.size(); i++) {
        if (axis == order[i]) {
            return i;
        }
    }
    return -1;
}

void MKLDNNConcatNode::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    if (!isOptimized()) {
        auto config = selected_pd->getConfig();
        if (!isInitConfig(config)) {
            for (size_t i = 0; i < config.inConfs.size(); i++) {
                config.inConfs[i].desc = getConfiguredInputDesc(config, i);
                // Concat doesn't support different precision on inputs
                config.inConfs[i].desc.setPrecision(inputPrecision);
            }

            for (size_t i = 0; i < config.outConfs.size(); i++) {
                config.outConfs[i].desc = getConfiguredOutputDesc(config, i);
                config.outConfs[i].desc.setPrecision(outputPrecision);
            }

            initDescriptor(config);
        }

        return;
    }

    auto config = selected_pd->getConfig();
    if (isInitConfig(config))
        return;

    for (size_t i = 0; i < config.outConfs.size(); i++) {
        if (!isUninitTensorDesc(config.outConfs[i].desc))
            continue;

        int num = getChildEdgeAt(i)->getOutputNum();
        if (num >= 0) {
            auto childConf = getChildEdgeAt(i)->getChild()->getSelectedPrimitiveDescriptor()->getConfig().inConfs[num];
            childConf.desc.setPrecision(config.outConfs[i].desc.getPrecision());

            if (getChildEdgeAt(i)->getChild()->getSelectedPrimitiveDescriptor()) {
                if (isUninitTensorDesc(childConf.desc) && childConf.inPlace >= 0)
                    getChildEdgeAt(i)->getChild()->initOptimalPrimitiveDescriptor();

                if (!isUninitTensorDesc(childConf.desc) &&
                        MKLDNNExtensionUtils::initTensorsAreEqual(childConf.desc, config.outConfs[i].desc)) {
                    config.outConfs[i].desc = childConf.desc;
                    continue;
                }
            }
        }
        config.outConfs[i].desc = InferenceEngine::TensorDesc(config.outConfs[i].desc.getPrecision(),
                                                              config.outConfs[i].desc.getDims(), {
                                                                      config.outConfs[i].desc.getBlockingDesc().getBlockDims(),
                                                                      config.outConfs[i].desc.getBlockingDesc().getOrder()
                                                              });
    }
    size_t offset = 0;
    for (size_t i = 0; i < config.inConfs.size(); i++) {
        config.inConfs[i].desc = InferenceEngine::TensorDesc(config.inConfs[i].desc.getPrecision(),
                                                             config.inConfs[i].desc.getDims(), {
                                                                  config.inConfs[i].desc.getBlockingDesc().getBlockDims(),
                                                                  config.inConfs[i].desc.getBlockingDesc().getOrder(),
                                                                  config.outConfs[0].desc.getBlockingDesc().getOffsetPadding() + offset,
                                                                  config.outConfs[0].desc.getBlockingDesc().getOffsetPaddingToData(),
                                                                  config.outConfs[0].desc.getBlockingDesc().getStrides()
                                                             });
        size_t axisSize = 1;

        if (config.inConfs[0].desc.getLayout() == Layout::NHWC) {
            // This is more general and works for any "direct" Layout (such as nchw or nhwc), but it doesn't work for nchw8c
            size_t realAxis = inverseOrder(config.inConfs[0].desc.getBlockingDesc().getOrder(), axis);
            for (size_t j = realAxis; j < config.inConfs[i].desc.getBlockingDesc().getBlockDims().size(); j++) {
                size_t jj = config.inConfs[0].desc.getBlockingDesc().getOrder()[j];
                axisSize *= config.inConfs[i].desc.getBlockingDesc().getBlockDims()[jj];
            }
        } else {
            // This works for nchw and nchw8c/nchw16c
            for (size_t j = axis; j < config.inConfs[i].desc.getBlockingDesc().getBlockDims().size(); j++) {
                axisSize *= config.inConfs[i].desc.getBlockingDesc().getBlockDims()[j];
            }
        }
        offset += axisSize;
    }
    initDescriptor(config);
}

void MKLDNNConcatNode::execute(mkldnn::stream strm) {
    if (isOptimized()) {
        return;
    }

    if (canOptimizeNspc) {
        execNspcSpecCase();
        return;
    }

    const MKLDNNMemory& dst_memory = getChildEdgeAt(0)->getMemory();
    const size_t num_src = getParentEdges().size();
    std::unordered_map<int, memory> mem_ags {{DNNL_ARG_DST, dst_memory.GetPrimitive()}};
    for (int i = 0; i < num_src; i++)
        mem_ags[DNNL_ARG_MULTIPLE_SRC + i] = getParentEdgeAt(i)->getMemory().GetPrimitive();

    (*prim).execute(strm, mem_ags);
}

InferenceEngine::Precision MKLDNNConcatNode::getRuntimePrecision() const {
    return MKLDNNExtensionUtils::getMaxPrecision(getInputPrecisions());
}

void MKLDNNConcatNode::execNspcSpecCase() {
    const MKLDNNMemory& dst_memory = getChildEdgeAt(0)->getMemory();
    const size_t num_src = getParentEdges().size();
    uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst_memory.GetData());
    const size_t dataSize = MKLDNNExtensionUtils::sizeOfDataType(dst_memory.GetDataType());

    std::vector<size_t> channelsDataSize;
    size_t channels_size = 0;
    std::vector<const uint8_t*> src_ptrs;
    std::vector<uint8_t*> dst_ptrs;

    for (size_t i = 0; i < num_src; i++) {
        const MKLDNNMemory& src_mem = getParentEdgeAt(i)->getMemory();
        const size_t num_channels = src_mem.GetDims()[channelAxis];

        channelsDataSize.push_back(num_channels * dataSize);
        src_ptrs.push_back(reinterpret_cast<const uint8_t*>(src_mem.GetData()));
        dst_ptrs.push_back(dst_ptr + channels_size);
        channels_size += num_channels * dataSize;
    }

    const size_t iter_count = getParentEdgeAt(0)->getMemory().GetSize() / channelsDataSize[0];

    parallel_for(iter_count, [&](int i) {
        const size_t dst_off = i * channels_size;
        for (int j = 0; j < num_src; j++) {
            cpu_memcpy(dst_ptrs[j] + dst_off, src_ptrs[j] + i * channelsDataSize[j], channelsDataSize[j]);
        }
    });
}

REG_MKLDNN_PRIM_FOR(MKLDNNConcatNode, Concatenation);
