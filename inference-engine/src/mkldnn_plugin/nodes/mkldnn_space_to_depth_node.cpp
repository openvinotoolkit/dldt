// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_space_to_depth_node.h"

#include <legacy/ie_layers.h>
#include <ie_parallel.hpp>
#include <ie_common.h>

#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "common/cpu_memcpy.h"

#include <string>
#include <cmath>

#define THROW_ERROR THROW_IE_EXCEPTION << "SpaceToDepth layer with name '" << getName() << "' "

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNSpaceToDepthNode::MKLDNNSpaceToDepthNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNSpaceToDepthNode::getSupportedDescriptors() {
    auto* spaceToDepthLayer = dynamic_cast<SpaceToDepthLayer*>(getCnnLayer().get());
    if (spaceToDepthLayer == nullptr)
        THROW_ERROR << "cannot convert from CNN layer";

    if (spaceToDepthLayer->insData[0].lock() == nullptr)
        THROW_ERROR << "has nullable input data";

    SizeVector srcDims = spaceToDepthLayer->insData[0].lock()->getTensorDesc().getDims();
    if (srcDims.size() < 3)
        THROW_ERROR << "has incorrect number of input dimensions";
    if (srcDims.size() > 5)
        THROW_ERROR << "doesn't support dimensions with rank greater than 5";

    if (spaceToDepthLayer->outData[0] == nullptr)
        THROW_ERROR << "has nullable output data";

    SizeVector dstDims = spaceToDepthLayer->outData[0]->getTensorDesc().getDims();
    if (srcDims.size() != dstDims.size())
        THROW_ERROR << "has incorrect number of input/output dimensions";

    std::string modeString = spaceToDepthLayer->GetParamAsString("mode");
    if (modeString == "blocks_first") {
        mode = Mode::BLOCKS_FIRST;
    } else if (modeString == "depth_first") {
        mode = Mode::DEPTH_FIRST;
    } else {
        THROW_ERROR << "doesn't support mode: " << modeString;
    }

    blockSize = spaceToDepthLayer->GetParamAsUInt("block_size", 1);
    if (blockSize == 0)
        THROW_ERROR << "has incorrect block_size parameter is zero!";

    size_t nSpatialDims = srcDims.size() - 2;
    blockStep = static_cast<size_t>(std::pow(blockSize, nSpatialDims));
    if (dstDims[1] % blockStep)
        THROW_ERROR << "has block_size parameter which is incompatible with output tensor channels dimension size";

    if (dstDims[1] / blockStep != srcDims[1])
        THROW_ERROR << "has incompatible input/output channels";

    for (size_t i = 0; i < nSpatialDims; ++i) {
        if (dstDims[i + 2] * blockSize != srcDims[i + 2])
            THROW_ERROR << "has incompatible spatial dims";
    }

    if (getParentEdges().size() != 1)
        THROW_ERROR << "has incorrect number of input edge";
    if (getChildEdges().empty())
        THROW_ERROR << "has incorrect number of output edges";
}

void MKLDNNSpaceToDepthNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    auto dataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto srcDims = getParentEdgeAt(0)->getDims();
    int nDims = srcDims.ToSizeVector().size();

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;

    auto pushSupportedPrimitiveDescriptor = [&](const mkldnn::memory::format_tag memoryFormat) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), dataType, memoryFormat);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), dataType, memoryFormat);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memoryFormat});
    };

    auto canUseBlocked = [=](const size_t block) {
        return mode == Mode::DEPTH_FIRST ? block % blockStep == 0 : true;
    };

    if (nDims == 4) {
        pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nhwc);
        if (srcDims[1] % 8 == 0 && canUseBlocked(8))
            pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nChw8c);
        if (srcDims[1] % 16 == 0 && canUseBlocked(16))
            pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nChw16c);
    } else if (nDims == 5) {
        pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::ndhwc);
        if (srcDims[1] % 8 == 0 && canUseBlocked(8))
            pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nCdhw8c);
        if (srcDims[1] % 16 == 0 && canUseBlocked(16))
            pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nCdhw16c);
    }
    pushSupportedPrimitiveDescriptor(MKLDNNMemory::GetPlainFormat(getParentEdgeAt(0)->getDims()));
}

void MKLDNNSpaceToDepthNode::createPrimitive() {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_ERROR << "has not allocated destination memory";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_ERROR << "has not allocated input memory";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << "has unidentified preferable primitive descriptor";

    SizeVector srcDims = getParentEdgeAt(0)->getBlob()->getTensorDesc().getDims();
    SizeVector dstDims = getChildEdgeAt(0)->getBlob()->getTensorDesc().getDims();
    prepareParams(srcDims, dstDims);

    size_t nDims = srcDims.size();
    const size_t nSpatialDims = nDims - 2;
    const bool isBlocked = getParentEdgeAt(0)->getMemory().GetDesc().isBlockedCFormat();
    nDims += nSpatialDims + static_cast<int>(isBlocked) + static_cast<int>(isBlocked && mode == Mode::DEPTH_FIRST);
    const size_t lastIdx = nDims - 1;

    order.resize(nDims);
    optimizedParams.src_block_dims.resize(nDims);
    order[0] = 0;
    optimizedParams.src_block_dims[0] = srcDims[0];

    auto orderAndReshape = [&](const size_t idx1, const size_t idx2, const size_t shift, const SizeVector& dims) {
        for (size_t i = 0; i < nSpatialDims; i++) {
            order[i + idx1] = i * 2 + shift;
            order[i + idx2] = i * 2 + shift + 1;

            optimizedParams.src_block_dims[order[i + idx1]] = dims[i + shift];
            optimizedParams.src_block_dims[order[i + idx2]] = blockSize;
        }
    };

    if (isBlocked) {
        SizeVector srcBlockedDims = getParentEdgeAt(0)->getDesc().getBlockingDesc().getBlockDims();
        SizeVector dstBlockedDims = getChildEdgeAt(0)->getDesc().getBlockingDesc().getBlockDims();

        size_t orderShiftForBlocks, orderShiftForDims;
        if (mode == Mode::BLOCKS_FIRST) {
            orderShiftForBlocks = nSpatialDims + 2;
            orderShiftForDims = 1;

            order[nSpatialDims + 1] = 1;
            order[lastIdx] = lastIdx;

            optimizedParams.src_block_dims[order[nSpatialDims + 1]] = srcBlockedDims[1];
            optimizedParams.src_block_dims[order[lastIdx]] = srcBlockedDims.back();
        } else {
            orderShiftForBlocks = 3;
            orderShiftForDims = nSpatialDims + 4;

            size_t extraBlockSize = srcBlockedDims.back() / blockStep;
            optimizedParams.src_block_dims[1] = srcBlockedDims[1];
            optimizedParams.src_block_dims[lastIdx] = extraBlockSize;
            optimizedParams.src_block_dims[lastIdx - 1] = blockStep;

            order[1] = 1;
            order[2] = lastIdx - 1;
            order[lastIdx - nSpatialDims] = lastIdx;
        }

        orderAndReshape(orderShiftForBlocks, orderShiftForDims, 2, dstBlockedDims);
    } else if (getParentEdgeAt(0)->getMemory().GetDesc().isTailCFormat()) {
        srcDims.push_back(srcDims[1]);
        dstDims.push_back(dstDims[1]);
        srcDims.erase(srcDims.begin() + 1);
        dstDims.erase(dstDims.begin() + 1);

        size_t shift = static_cast<size_t>(mode == DEPTH_FIRST) + nSpatialDims + 1;
        order[mode == Mode::DEPTH_FIRST ? nSpatialDims + 1 : lastIdx] = lastIdx;
        optimizedParams.src_block_dims[lastIdx] = srcDims.back();

        orderAndReshape(1, shift, 1, dstDims);
    } else {
        size_t shift = static_cast<size_t>(mode == DEPTH_FIRST) + 1;
        order[mode == Mode::DEPTH_FIRST ? 1 : nSpatialDims + 1] = 1;
        optimizedParams.src_block_dims[1] = srcDims[1];

        orderAndReshape(nSpatialDims + 2, shift, 2, dstDims);
    }

    prepareOptimizedParams(nDims, getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getPrecision());
    prepareConfigParams();
}

void MKLDNNSpaceToDepthNode::execute(mkldnn::stream strm) {
    auto srcData = reinterpret_cast<const uint8_t*>(this->getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto dstData = reinterpret_cast<uint8_t*>(this->getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    int MB = batchToProcess();
    if (params.shape5D[0] != MB)
        params.shape5D[0] = MB;

    if (permute_kernel) {
        auto &jcp = (*permute_kernel).jcp;
        if (jcp.dst_block_dims[0] != MB)
            jcp.dst_block_dims[0] = MB;

        optimizedExecute(srcData, dstData);
        return;
    }

    if (getParentEdgeAt(0)->getMemory().GetDesc().isBlockedCFormat()) {
        size_t srcCountBlocks = this->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getBlockingDesc().getBlockDims()[1];
        size_t block = this->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getBlockingDesc().getBlockDims().back();
        size_t blockRemainder = params.srcChannels % block;
        size_t lastBlock = blockRemainder == 0 ? block : blockRemainder;

        size_t dstBlock = block * params.shape5D[2] * params.shape5D[3] * params.shape5D[4];
        size_t srcBlock = block * params.shape5D[2] * params.block3D[0] * params.shape5D[3] * params.block3D[1] *
                          params.shape5D[4] * params.block3D[2];

        parallel_for2d(params.shape5D[0], params.shape5D[2], [&](size_t i0, size_t i2) {
            size_t srcIdx1 = i0 * srcBlock * srcCountBlocks;
            size_t dstIdx1 = i0 * params.batchStep + i2 * params.shape5D[3] * params.shape5D[4] * block;
            for (size_t b2 = 0; b2 < params.block3D[0]; ++b2) {
                size_t blk2 = (b2 * params.block3D[1] * params.block3D[2] * params.blockShift);
                size_t blockNum2 = blk2 / block;
                size_t blockRemainder2 = blk2 - blockNum2 * block;

                size_t srcIdx2 = srcIdx1 + (i2 * params.block3D[0] + b2) * params.shape5D[3] * params.block3D[1] *
                                 params.shape5D[4] * params.block3D[2] * block;
                size_t dstIdx2 = dstIdx1 + blockNum2 * dstBlock;
                for (size_t b3 = 0; b3 < params.block3D[1]; ++b3) {
                    size_t blk3 = (blockRemainder2 + b3 * params.blockShift * params.block3D[2]);
                    size_t blockNum3 = blk3 / block;
                    size_t blockRemainder3 = blk3 - blockNum3 * block;

                    for (size_t i3 = 0; i3 < params.shape5D[3]; ++i3) {
                        size_t srcIdx3 = srcIdx2 + (i3 * params.block3D[1] + b3) * params.shape5D[4] * params.block3D[2] * block;
                        size_t dstIdx3 = dstIdx2 + i3 * params.shape5D[4] * block + blockNum3 * dstBlock;
                        for (size_t b4 = 0; b4 < params.block3D[2]; ++b4) {
                            size_t blk4 = (blockRemainder3 + b4 * params.blockShift);
                            size_t blockNum4 = blk4 / block;
                            size_t blockRemainder4 = blk4 - blockNum4 * block;

                            for (size_t i4 = 0; i4 < params.shape5D[4]; ++i4) {
                                size_t srcIdx4 = srcIdx3 + (i4 * params.block3D[2] + b4) * block;
                                size_t dstIdx4 = dstIdx3 + i4 * block + blockNum4 * dstBlock;
                                for (size_t i5 = 0; i5 < srcCountBlocks; ++i5) {
                                    size_t size = (i5 == srcCountBlocks - 1) ? lastBlock : block;
                                    for (size_t i6 = 0; i6 < size; ++i6) {
                                        size_t blk5 = blockRemainder4 + (i6 + i5 * block) * params.channelShift;
                                        size_t blockNum5 = blk5 / block;
                                        size_t blockRemainder5 = blk5 - blockNum5 * block;

                                        size_t dstIdx5 = dstIdx4 + blockRemainder5 + blockNum5 * dstBlock;
                                        size_t srcIdx5 = srcIdx4 + i6 + i5 * srcBlock;
                                        cpu_memcpy(dstData + dstIdx5 * optimizedParams.data_size,
                                                   srcData + srcIdx5 * optimizedParams.data_size,
                                                   optimizedParams.data_size);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    } else if (getParentEdgeAt(0)->getMemory().GetDesc().isTailCFormat()) {
        parallel_for2d(params.shape5D[0], params.shape5D[2], [&](size_t i0, size_t i2) {
            size_t srcIdx1 = i0 * params.batchStep;
            size_t dstIdx1 = i0 * params.batchStep;
            for (size_t b2 = 0; b2 < params.block3D[0]; b2++) {
                size_t srcIdx2 = srcIdx1 + (i2 * params.block3D[0] + b2) * params.shape5D[3] * params.block3D[1] *
                                 params.shape5D[4] * params.block3D[2] * params.srcChannels;
                size_t dstIdx2 = dstIdx1 + i2 * params.shape5D[3] * params.shape5D[4] * params.dstChannels +
                                 b2 * params.block3D[1] * params.block3D[2] * params.blockShift;
                for (size_t i3 = 0; i3 < params.shape5D[3]; i3++) {
                    for (size_t b3 = 0; b3 < params.block3D[1]; b3++) {
                        size_t dstIdx3 = dstIdx2 + i3 * params.shape5D[4] * params.dstChannels + b3 * params.block3D[2] * params.blockShift;
                        size_t srcIdx3 = srcIdx2 + (i3 * params.block3D[1] + b3) * params.shape5D[4] * params.block3D[2] * params.srcChannels;
                        for (size_t i4 = 0; i4 < params.shape5D[4]; i4++) {
                            for (size_t b4 = 0; b4 < params.block3D[2]; b4++) {
                                size_t srcIdx4 = srcIdx3 + (i4 * params.block3D[2] + b4) * params.srcChannels;
                                size_t dstIdx4 = dstIdx3 + i4 * params.dstChannels + b4 * params.blockShift;
                                for (size_t i1 = 0; i1 < params.srcChannels; i1++) {
                                    size_t srcIdx5 = srcIdx4 + i1;
                                    size_t dstIdx5 = dstIdx4 + i1 * params.channelShift;
                                    cpu_memcpy(dstData + dstIdx5 * optimizedParams.data_size,
                                               srcData + srcIdx5 * optimizedParams.data_size,
                                               optimizedParams.data_size);
                                }
                            }
                        }
                    }
                }
            }
        });
    } else {
        parallel_for2d(params.shape5D[0], params.srcChannels, [&](size_t i0, size_t i1) {
            size_t srcIdx1 = i0 * params.batchStep + i1 * blockStep * params.spatialStep;
            size_t dstIdx1 = i0 * params.batchStep + i1 * params.channelShift * params.spatialStep;
            for (size_t i2 = 0; i2 < params.shape5D[2]; i2++) {
                for (size_t b2 = 0; b2 < params.block3D[0]; b2++) {
                    size_t srcIdx2 = srcIdx1 + (i2 * params.block3D[0] + b2) * params.shape5D[3] * params.block3D[1] * params.shape5D[4] * params.block3D[2];
                    size_t dstIdx2 = dstIdx1 + i2 * params.shape5D[3] * params.shape5D[4] +
                                     b2 * params.block3D[1] * params.block3D[2] * params.blockShift * params.spatialStep;
                    for (size_t i3 = 0; i3 < params.shape5D[3]; i3++) {
                        for (size_t b3 = 0; b3 < params.block3D[1]; b3++) {
                            size_t srcIdx3 = srcIdx2 + (i3 * params.block3D[1] + b3) * params.shape5D[4] * params.block3D[2];
                            size_t dstIdx3 = dstIdx2 + i3 * params.shape5D[4] + b3 * params.block3D[2] * params.blockShift * params.spatialStep;
                            for (size_t i4 = 0; i4 < params.shape5D[4]; i4++) {
                                for (size_t b4 = 0; b4 < params.block3D[2]; b4++) {
                                    size_t srcIdx4 = srcIdx3 + i4 * params.block3D[2] + b4;
                                    size_t dstIdx4 = dstIdx3 + i4 + b4 * params.blockShift * params.spatialStep;
                                    cpu_memcpy(dstData + dstIdx4 * optimizedParams.data_size,
                                               srcData + srcIdx4 * optimizedParams.data_size,
                                               optimizedParams.data_size);
                                }
                            }
                        }
                    }
                }
            }
        });
    }
}

bool MKLDNNSpaceToDepthNode::created() const {
    return getType() == SpaceToDepth;
}
REG_MKLDNN_PRIM_FOR(MKLDNNSpaceToDepthNode, SpaceToDepth);
