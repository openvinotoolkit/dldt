// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "base.hpp"

#include <string>
#include <vector>

#include <ngraph/op/ctc_greedy_decoder_seq_len.hpp>
#include "ie_parallel.hpp"
#include "mkldnn_ctc_greedy_decoder_seq_len_node.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNCTCGreedyDecoderSeqLenNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto greedyDecOp = ngraph::as_type_ptr<const ngraph::op::v6::CTCGreedyDecoderSeqLen>(op);
        if (!greedyDecOp) {
            errorMessage = "Node is not an instance of the CTCGreedyDecoderSeqLen operation from operation set v6.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNCTCGreedyDecoderSeqLenNode::MKLDNNCTCGreedyDecoderSeqLenNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    std::string errPrefix = "CTCGreedyDecoderSeqLen layer with name '" + op->get_friendly_name() + "' ";
    if (op->get_input_size() < 2 || op->get_input_size() > 3)
        IE_THROW() << errPrefix << "has invalid number of input edges: " << op->get_input_size();
    if (op->get_output_size() != 2)
        IE_THROW() << errPrefix << "has invalid number of outputs edges: " << op->get_output_size();

    if (op->get_input_shape(DATA_INDEX)[0] != op->get_input_shape(SEQUENCE_LENGTH_INDEX)[0])
        IE_THROW() << errPrefix << "has invalid input shapes.";

    Precision inDataPrecision = details::convertPrecision(op->get_input_element_type(DATA_INDEX));
    if (inDataPrecision != Precision::FP32 && inDataPrecision != Precision::BF16)
        IE_THROW() << errPrefix << "has unsupported 'data' input precision: " << inDataPrecision;

    Precision seqLenPrecision = details::convertPrecision(op->get_input_element_type(SEQUENCE_LENGTH_INDEX));
    if (seqLenPrecision != Precision::I32 && seqLenPrecision != Precision::I64)
        IE_THROW() << errPrefix << "has unsupported 'sequence_length' input precision: " << seqLenPrecision;

    auto greedyDecOp = ngraph::as_type_ptr<const ngraph::op::v6::CTCGreedyDecoderSeqLen>(op);
    mergeRepeated = greedyDecOp->get_merge_repeated();

    size_t sizeVector = op->get_input_size();
    inDataConf.reserve(sizeVector);
    inDataConf.emplace_back(TensorDescCreatorTypes::ncsp, Precision::FP32);
    for (int i = 1; i < sizeVector; ++i)
        inDataConf.emplace_back(TensorDescCreatorTypes::ncsp, Precision::I32);
}

void MKLDNNCTCGreedyDecoderSeqLenNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    addSupportedPrimDesc(inDataConf,
                         {{TensorDescCreatorTypes::ncsp, Precision::I32},
                          {TensorDescCreatorTypes::ncsp, Precision::I32}},
                         impl_desc_type::ref_any);
}

void MKLDNNCTCGreedyDecoderSeqLenNode::execute(mkldnn::stream strm) {
    const float* probabilities = reinterpret_cast<const float  *>(getParentEdgeAt(DATA_INDEX)->getMemoryPtr()->GetPtr());
    const int* sequenceLengths = reinterpret_cast<const int  *>(getParentEdgeAt(SEQUENCE_LENGTH_INDEX)->getMemoryPtr()->GetPtr());
    int* decodedClasses =  reinterpret_cast<int  *>(getChildEdgeAt(DECODED_CLASSES_INDEX)->getMemoryPtr()->GetPtr());
    int* decodedClassesLength = reinterpret_cast<int  *>(getChildEdgeAt(DECODED_CLASSES_LENGTH_INDEX)->getMemoryPtr()->GetPtr());

    const size_t B = getParentEdgeAt(DATA_INDEX)->getDims()[0];;
    const size_t T = getParentEdgeAt(DATA_INDEX)->getDims()[1];;
    const int C = getParentEdgeAt(DATA_INDEX)->getDims()[2];;
    const size_t TC = T * C;

    int blankIndex = C - 1;
    if (getParentEdges().size() > BLANK_INDEX)
        blankIndex = (reinterpret_cast<const int  *>(getParentEdgeAt(BLANK_INDEX)->getMemoryPtr()->GetPtr()))[0];

    size_t workAmount = 0;
    for (size_t b = 0; b < B; b++) {
        if (sequenceLengths[b] > T) {
            std::string errorMsg = errorPrefix
                                   + ". Sequence length " + std::to_string(sequenceLengths[b])
                                   + " cannot be greater than according decoded classes dimension size "
                                   + std::to_string(getChildEdgeAt(DECODED_CLASSES_INDEX)->getDims()[1]);
            IE_THROW() << errorMsg;
        }
        workAmount += sequenceLengths[b];
    }
    // Parallelization could not be made directly by T due to output index depends on merged classes and
    // blank index, thus could not be shared between threads. Better to divide operation on two steps.
    // At the first stage find the maximum index. At second stage merge if needed.
    // Such approach makes parallelization more efficient.
    auto threadBody = [&](const int ithr, const int nthr) {
        size_t start(0lu), end(0lu);
        splitter(workAmount, nthr, ithr, start, end);
        if (start >= end)
            return;
        size_t tStart = 0lu, bStart = 0lu;
        for (; bStart < B; bStart++) {
            tStart += sequenceLengths[bStart];
            if (tStart >= start) {
                tStart = start - (tStart - sequenceLengths[bStart]);
                break;
            }
        }

        size_t workCounter = start;

        for (size_t b = bStart; b < B; ++b) {
            size_t outputIndex = b * T + tStart;
            const float* probs = probabilities + b * TC + C * tStart;
            const size_t actualSeqLen = sequenceLengths[b];

            for (size_t t = tStart; t < actualSeqLen; ++t) {
                int maxClassIdx = 0;
                float maxProb = probs[0];
                probs++;

                for (int c = 1; c < C; c++, probs++) {
                    if (*probs > maxProb) {
                        maxClassIdx = c;
                        maxProb = *probs;
                    }
                }
                decodedClasses[outputIndex++] = maxClassIdx;

                if (++workCounter >= end) {
                    return;
                }
            }
            tStart = 0lu;
        }
    }; // thread body

    parallel_nt(0, threadBody);

    parallel_for(B, [&](size_t b) {
        int prevClassIdx = -1;
        size_t outputIndex = b * T;
        const size_t actualSeqLen = sequenceLengths[b];
        int* shiftedOut = decodedClasses + b * T;

        for (size_t t = 0; t < actualSeqLen; ++t) {
            if (*shiftedOut != blankIndex &&
                !(mergeRepeated && *shiftedOut == prevClassIdx)) {
                decodedClasses[outputIndex++] = *shiftedOut;
            }
            prevClassIdx = *shiftedOut;
            shiftedOut++;
        }
        std::fill(decodedClasses + outputIndex, decodedClasses + (b + 1) * T, -1);
        decodedClassesLength[b] = outputIndex - b * T;
    });
}

bool MKLDNNCTCGreedyDecoderSeqLenNode::created() const {
    return getType() == CTCGreedyDecoderSeqLen;
}

REG_MKLDNN_PRIM_FOR(MKLDNNCTCGreedyDecoderSeqLenNode, CTCGreedyDecoderSeqLen)
