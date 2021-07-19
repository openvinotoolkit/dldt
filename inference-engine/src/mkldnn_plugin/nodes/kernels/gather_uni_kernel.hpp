// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../base.hpp"
#include "cpu/x64/jit_generator.hpp"
#include <mkldnn_types.h>

namespace MKLDNNPlugin {

struct jGatherConfParams {
    int32_t beforeAxisSize;
    int32_t indicesSize;
    int32_t axisDim;
    uint32_t dataTypeSize;
    int64_t afterAxisSize;
    bool blockedIndices512 = false;
    bool blockedIndices256 = false;
    bool blockedIndices128 = false;
};

struct gatherJitExecArgs {
    const void* src;
    void* dst;
    const int* indices;
    const int* dataTypeSize;
    const int* idxTypeSize;
    const int* axisDim;
    const int* axDimSum;
    const int* shufMask8bitUni;
    const int* permMask8bitA2;
    const int* permMask8bitA5;
    const int* shufMask16bitUni;
    const int* permMask16bitA2;
    const int* permMask16bitA5;
    const int* minusOne;
    const int* batchIndices;
    const int* specIndices;
    const int* specIndicesSizePtr;
    const int* betweenBatchAndAxisIdx;
    size_t     betweenBatchAndAxisSize;
    const int* axisAndAfterAxisSize;
    const int* srcAfterBatchSizeInBytes;
    const uint32_t* vecLen;
    const int* permIdx;
    const int* beforeAxisDiff;
    size_t idxIter = 0;
    size_t idxStartB = 0;
    size_t workAmount = 0;
    size_t afterAxisBlockSize = 0;
    size_t beforeAxisCounter = 0;
    size_t specIndicesSizeInBytes = 0;
    int* tmp; // remove
    int* retVal; // remove
};

struct jitGatherKernelBase {
    void (*ker_)(const gatherJitExecArgs *);
    void operator()(const gatherJitExecArgs *args) {
        assert(ker_);
        ker_(args);
    }
    explicit jitGatherKernelBase(jGatherConfParams jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jitGatherKernelBase() {}

    virtual void create_ker() = 0;
    inline uint32_t getVecLen() {
        return vlen;
    }
    inline uint32_t geElPerVec() {
        return elPerVec;
    }

protected:
    jGatherConfParams jcp_;
    uint32_t vlen;
    uint32_t elPerVec;
};

template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
struct jitUniGatherKernel : public jitGatherKernelBase, public mkldnn::impl::cpu::x64::jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jitUniGatherKernel)

    explicit jitUniGatherKernel(jGatherConfParams jcp);

    void create_ker() override;
    void generate() override;

protected:
    using Vmm = typename mkldnn::impl::utils::conditional3<isa == mkldnn::impl::cpu::x64::sse41, Xbyak::Xmm,
            isa == mkldnn::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    const uint32_t vlenXmm = mkldnn::impl::cpu::x64::cpu_isa_traits<mkldnn::impl::cpu::x64::sse41>::vlen;
    const uint32_t vlenYmm = mkldnn::impl::cpu::x64::cpu_isa_traits<mkldnn::impl::cpu::x64::avx2>::vlen;
    uint32_t dataTypeShift = 0;

    Xbyak::Reg64 regSrc = r8;
    Xbyak::Reg64 regDst = r9;
    Xbyak::Reg64 regIndices = r10;
    Xbyak::Reg64 regIdxIter = r11;
    Xbyak::Reg64 regWorkAmount = r12;
    Xbyak::Reg64 regAux1 = r13;
//    Xbyak::Reg64 regAux2 = r14;
    Xbyak::Reg64 regSpecIdxSizeInBytes = r14;
    Xbyak::Reg64 regAux3 = r15;
    Xbyak::Reg64 regBetweenBatchAndAxisIter = r15;
    Xbyak::Reg64 regBetweenBatchAndAxisSize = rbx;
//    Xbyak::Reg64 regSpecIndicesSize = rbp;

    Xbyak::Reg64 regParams = mkldnn::impl::cpu::x64::abi_param1;

    Xbyak::Opmask kMaskOnes = Xbyak::Opmask(1);
    Xbyak::Opmask kMaskAux1 = Xbyak::Opmask(2);
    Xbyak::Opmask kMaskAux2 = Xbyak::Opmask(3);
    Xbyak::Opmask kGatherMask = Xbyak::Opmask(4);

    Xbyak::Xmm xmmAux0 = Xbyak::Xmm(0);
    Xbyak::Xmm xmmAux1 = Xbyak::Xmm(1);
    Xbyak::Xmm xmmAxDimSum = Xbyak::Xmm(2);
    Xbyak::Xmm xmmAxDim = Xbyak::Xmm(3);
    Xbyak::Xmm xmmDictTypeSize = Xbyak::Xmm(4);
    Xbyak::Xmm xmmSrcShifts = Xbyak::Xmm(5);
    Xbyak::Xmm xmmMinusOne = Xbyak::Xmm(6);
    Xbyak::Xmm xmmAux2 = Xbyak::Xmm(7);
    Xbyak::Xmm xmmAux3 = Xbyak::Xmm(8);
    Xbyak::Xmm xmmAux9 = Xbyak::Xmm(9);
    Xbyak::Xmm xmmAux7 = Xbyak::Xmm(12);
    Xbyak::Xmm xmmOnes = Xbyak::Xmm(13);
    Xbyak::Xmm xmmDst = Xbyak::Xmm(15);

    Xbyak::Ymm ymmAux0 = Xbyak::Ymm(0);
    Xbyak::Ymm ymmAux1 = Xbyak::Ymm(1);
    Xbyak::Ymm ymmAux2 = Xbyak::Ymm(7);
    Xbyak::Ymm ymmAux10 = Xbyak::Ymm(16);

    Xbyak::Zmm zmmAux0 = Xbyak::Zmm(0);
    Xbyak::Zmm zmmAux1 = Xbyak::Zmm(1);
    Xbyak::Zmm zmmSrcShifts = Xbyak::Zmm(5);
    Xbyak::Zmm zmmDst = Xbyak::Zmm(15);

    Vmm vmmAux0 = Vmm(0);
    Vmm vmmAux1 = Vmm(1);
//    Vmm vmmAxDimSum = Vmm(2);
    Vmm vmmAxisAndAfterAxisSize = Vmm(2);
    Vmm vmmSrcAfterBatchSize = Vmm(2);
//    Vmm vmmAxDim = Vmm(3);
    Vmm vmmBeforeAxisSum = Vmm(3);
//    Vmm vmmDictTypeSize = Vmm(4);
    Vmm vmmAux4 = Vmm(4);
    Vmm vmmSrcShifts = Vmm(5);
//    Vmm vmmMinusOne = Vmm(6);
    Vmm vmmZeros = Vmm(6);
    Vmm vmmVecLen = Vmm(7);
    Vmm vmmAux8 = Vmm(7);
    Vmm vmmAux3 = Vmm(8);
    Vmm vmmPermIdx = Vmm(8);
//    Vmm vmmAux4 = Vmm(9);
    Vmm vmmSpecIndices = Vmm(9);
    Vmm vmmAux5 = Vmm(10);
    Vmm vmmIdxBatchSum = Vmm(10);
    Vmm vmmBeforeAxisDiff = Vmm(10);
    Vmm vmmGatherMask = Vmm(11);
    Vmm vmmSpecIdxSize = Vmm(12);
//    Vmm vmmAux8 = Vmm(13);
    Vmm vmmOnes = Vmm(13);
    Vmm vmmAux9 = Vmm(14);
    Vmm vmmAxisDim = Vmm(14);
    Vmm vmmDst = Vmm(15);

    Vmm vmmAux11 = Vmm(17);

    void calcSrcShiftLong(Xbyak::Ymm& dstIndices, Xbyak::Ymm& dstMask, Xbyak::Ymm& idxMask); // remove idxMask ?
    void calcSrcShiftLong(Xbyak::Zmm& dstIndices, Xbyak::Opmask& dstMask, Xbyak::Opmask& idxMask);
    void calcSrcShiftShort(Xbyak::Ymm& dst, Xbyak::Ymm& dstMask);
    void calcSrcShiftShort(Xbyak::Zmm& dst, Xbyak::Opmask& dstMask);
    void normalizeRawIndices(Xbyak::Ymm& rawIndices, Xbyak::Ymm& dstMask, Xbyak::Ymm& aux);
    void normalizeRawIndices(Xbyak::Zmm& rawIndices, Xbyak::Opmask& dstMask, Xbyak::Opmask& auxMask);
    void gatherLongIdx32();
    void gatherShortIdx32();
    void gatherLongIdx16();
    void gatherShortIdx16();
    void gatherLongIdx8();
    void gatherShortIdx8();
    void tail32();
    void tail16();
    void tail8();
};

}  // namespace MKLDNNPlugin
