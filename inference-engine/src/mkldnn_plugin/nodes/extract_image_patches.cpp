// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extract_image_patches.hpp"
#include "list.hpp"
#include <cpu/x64/jit_generator.hpp>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using details::CaselessEq;

using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_eximpat_args, field)

template <cpu_isa_t isa>
struct jit_uni_eximpat_kernel_f32 : public jit_uni_eximpat_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_eximpat_kernel_f32)

    explicit jit_uni_eximpat_kernel_f32(jit_eximpat_params jpp) : jit_uni_eximpat_kernel(jpp), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        this->preamble();

        mov(reg_src_oh, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        loop();  //loop over input and output to get the work done

        this->postamble();
    }

private:
    using Vmm = typename conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using reg16_t = const Xbyak::Reg16;
    using reg8_t = const Xbyak::Reg8;
    uint32_t vlen = cpu_isa_traits<isa>::vlen;

    reg64_t reg_src_oh = r8;
    reg64_t reg_dst = r9;

    reg64_t reg_oh_count = r10;
    reg64_t reg_ow_count = r11;
    reg64_t reg_num_pads = r12; // reserved. used to specify padding
    reg64_t reg_src_ow = r13;
    reg64_t reg_aux64 = rax;
//    reg32_t reg_aux32 = r13d;
//    reg16_t reg_aux16 = r13w;
//    reg8_t reg_aux8 = r13b;
    //reg64_t reg_aux64_2 = r14;
    //reg32_t reg_aux32_2 = r14d;

    reg64_t reg_params = abi_param1;

    Vmm vmm = Vmm(0);
    Xmm xmm = Xmm(0);
    Vmm vmm_zero = Vmm(1); // reserved for pad
    Xbyak::Xmm xmm_aux = Xbyak::Xmm(2);


    inline void store_vector(const Xbyak::Address &op, Vmm vmm_dst, Precision prec) {
        uni_vmovups(op, vmm_dst);
    }

    inline void load_scalar(Vmm vmm_arg, const Xbyak::Address &op) {
        Xbyak::Xmm xmm_src = Xmm(vmm_arg.getIdx());
        switch (jpp.dtype_size) {
            case 4: movss(xmm_src, op); break;
            case 2: pinsrw(xmm_src, op, 0x0); break;
            case 1: pinsrb(xmm_src, op, 0x0); break;
            default:
                assert(!"unknown dtype size");
        }
    }
    inline void store_scalar(const Xbyak::Address &op, Vmm vmm_arg) {
        Xbyak::Xmm xmm_dst = Xmm(vmm_arg.getIdx());
        switch (jpp.dtype_size) {
            case 4: movss(op, xmm_dst); break;
            case 2: pextrw(op, xmm_dst, 0x0); break;
            case 1: vpextrb(op, xmm_dst, 0x0); break;
            default:
                assert(!"unknown dtype size");
        }
    }

    void pad_with_zeros(reg64_t &reg_num_pads, reg64_t &reg_dst_arg) {
        Xbyak::Label main, tail, exit;
        L(main);
        {
            cmp(reg_num_pads, jpp.block_size);
            jl(tail);
            store_vector(ptr[reg_dst_arg], vmm_zero, jpp.precision);
            add(reg_dst_arg, jpp.dtype_size * jpp.block_size);
            sub(reg_num_pads, jpp.block_size);
            jmp(main);
        }
        L(tail);
        {
            cmp(reg_num_pads, 0);
            jle(exit);
            store_scalar(ptr[reg_dst_arg], vmm_zero);
            add(reg_dst_arg, jpp.dtype_size);
            sub(reg_num_pads, 1);
            jmp(tail);
        }
        L(exit);
    }

    void read_src2xmm(const Xbyak::Xmm &xmm_arg, reg64_t &reg_src_arg) {
        const int xmm_size = 16; // bytes
        const int xmm_block_size = xmm_size / jpp.dtype_size;
        for (int i = 0; i < xmm_block_size; i++) {
            Xbyak::Address addr = ptr[reg_src_arg + i * jpp.SW * jpp.dtype_size];
            switch (jpp.dtype_size) {
                case 8: uni_vpinsrq(xmm_arg, xmm_arg, addr, i); break;
                case 4: uni_vpinsrd(xmm_arg, xmm_arg, addr, i); break;
                case 2: uni_vpinsrw(xmm_arg, xmm_arg, addr, i); break;
                case 1: uni_vpinsrb(xmm_arg, xmm_arg, addr, i); break;
            }
        }
        add(reg_src_arg, jpp.SW * jpp.dtype_size * xmm_block_size);
    }
    void read_src2ymm(const Xbyak::Ymm &ymm_arg, reg64_t &reg_src_arg) {
        const int ymm_size = 32; // bytes
        const int ymm_block_size = ymm_size / jpp.dtype_size;
        for (int i = 0; i < ymm_block_size; i++) {
            Xbyak::Address addr = ptr[reg_src_arg + i * jpp.SW * jpp.dtype_size];
            switch (jpp.dtype_size) {
                case 8: uni_vpinsrq(ymm_arg, ymm_arg, addr, i); break;
                case 4: uni_vpinsrd(ymm_arg, ymm_arg, addr, i); break;
                case 2: uni_vpinsrw(ymm_arg, ymm_arg, addr, i); break;
                case 1: uni_vpinsrb(ymm_arg, ymm_arg, addr, i); break;
            }
        }
        add(reg_src_arg, jpp.SW * jpp.dtype_size * ymm_block_size);
    }

    void read_src2vmm(Vmm &vmm_arg, const Xbyak::Reg64 &reg_src_arg) {
        // isa == x64::sse41 || isa == x64::avx2 || isa == x64::avx512_common
        Xbyak::Xmm low_xmm = Xmm(vmm_arg.getIdx());
        read_src2xmm(low_xmm, reg_src_arg);
        if ((isa == x64::avx2) || (isa == x64::avx512_common)) {
            Xbyak::Ymm low_ymm = Xbyak::Ymm(vmm_arg.getIdx());
            read_src2xmm(xmm_aux, reg_src_arg);
            vinserti128(low_ymm, low_ymm, xmm_aux, 1);
            // NB! Need to implement further optimization for avx512?
            //if (isa == x64::avx512_common) {
            //}
        }
    }

    // The main loop where all the work is done
    void loop() {
        //for (int64_t i = 0; i < ih_lpad * OW; i++)
        //    dst_data[dst_idx++] = T(0);
        mov(reg_oh_count, ptr[reg_params + GET_OFF(h_hi_pad)]);
        mov(reg_num_pads, ptr[reg_params + GET_OFF(h_lo_pad)]);
        sub(reg_oh_count, reg_num_pads);
        mul_by_const(reg_num_pads, reg_aux64, jpp.OW);
        pad_with_zeros(reg_num_pads, reg_dst);
        /*
        // The whole loop:
        for (int64_t ishift = ioffset + ih_lpad * SH * IW; ishift < ioffset + ih_hpad * SH * IW; ishift += SH * IW) {
            for (int64_t iw = 0; iw < iw_lpad; iw++)
                dst_data[dst_idx++] = 0;
            for (int64_t src_idx = ishift + iw_lpad * SW; src_idx < ishift + iw_hpad * SW; src_idx += SW)
                dst_data[dst_idx++] = src_data[src_idx];

            for (int64_t i = 0; i < (OW - iw_hpad); i++)
                dst_data[dst_idx++] = 0;
        }
         */
        Xbyak::Label ih_loop, ih_tail, ih_exit;
        Xbyak::Label iw_loop, iw_tail, iw_exit;
        //for (int64_t ishift = ioffset + ih_lpad * SH * IW; ishift < ioffset + ih_hpad * SH * IW; ishift += SH * IW) {...}
        L(ih_loop);
        {
            cmp(reg_oh_count, 0);
            jle(ih_exit, T_NEAR);
            /*
            for (int64_t iw = 0; iw < iw_lpad; iw++)
                dst_data[dst_idx++] = 0;
            */
            mov(reg_ow_count, ptr[reg_params + GET_OFF(w_hi_pad)]);
            mov(reg_num_pads, ptr[reg_params + GET_OFF(w_lo_pad)]);
            mov(reg_src_ow, reg_num_pads);
            pad_with_zeros(reg_num_pads, reg_dst);
            /*
            for (int64_t src_idx = ishift + iw_lpad * SW; src_idx < ishift + iw_hpad * SW; src_idx += SW)
                dst_data[dst_idx++] = src_data[src_idx];
            */
            sub(reg_ow_count, reg_src_ow);
            mul_by_const(reg_src_ow, reg_aux64, jpp.SW * jpp.dtype_size);
            add(reg_src_ow, reg_src_oh);
            L(iw_loop);
            {
                cmp(reg_ow_count, jpp.block_size);
                jle(iw_tail, T_NEAR);
                read_src2vmm(vmm, reg_src_ow);
                store_vector(ptr[reg_dst], vmm, jpp.precision);
                add(reg_dst, jpp.dtype_size * jpp.block_size);
                sub(reg_ow_count, jpp.block_size);
                jmp(iw_loop);
            }
            L(iw_tail);
            {
                cmp(reg_ow_count, 0);
                jle(iw_exit, T_NEAR);
                load_scalar(vmm, ptr[reg_src_ow]);
                store_scalar(ptr[reg_dst], vmm);
                dec(reg_ow_count);
                add(reg_src_ow, jpp.SW * jpp.dtype_size);
                add(reg_dst, jpp.dtype_size);
                jmp(iw_tail);
            }
            L(iw_exit);
            /*
            for (int64_t i = 0; i < (OW - iw_hpad); i++)
                dst_data[dst_idx++] = 0;
            */
            mov(reg_num_pads, jpp.OW);
            sub(reg_num_pads, ptr[reg_params + GET_OFF(w_hi_pad)]);
            pad_with_zeros(reg_num_pads, reg_dst);
            dec(reg_oh_count);
            add(reg_src_oh, jpp.SH * jpp.IW * jpp.dtype_size);
            jmp(ih_loop, T_NEAR);
        }
        L(ih_exit);
        //for (int64_t i = 0; i < (OH - ih_hpad) * OW; i++)
        //    dst_data[dst_idx++] = T(0);
        mov(reg_num_pads, jpp.OH);
        sub(reg_num_pads, ptr[reg_params + GET_OFF(h_hi_pad)]);
        mul_by_const(reg_num_pads, reg_aux64, jpp.OW);
        pad_with_zeros(reg_num_pads, reg_dst);
    }
};

ExtractImagePatchesImpl::ExtractImagePatchesImpl(const CNNLayer* layer) {
    try {
        std::string errorPrefix = std::string("Layer ") + layer->type + " with name '" + layer->name + "' ";
        if (details::CaselessEq<std::string>()("ExtractImagePatchesLayer", layer->type))
            THROW_IE_EXCEPTION << errorPrefix << "is not an instance of ExtractImagePatchesLayer class";

        if (layer->insData.size() != 1 || layer->outData.size() != 1)
            THROW_IE_EXCEPTION << errorPrefix << "has incorrect number of input or output edges!"
                << " Input: " << layer->insData.size() << "; Output: " << layer->outData.size();

        auto inData = layer->insData[0].lock();
        if (inData == nullptr)
            THROW_IE_EXCEPTION << errorPrefix << "has nullable input data";

        if (inData->getTensorDesc().getDims().size() != 4)
            THROW_IE_EXCEPTION << errorPrefix << "must have 4D input tensor. Actual: " << inData->getTensorDesc().getDims().size();

        if (layer->outData[0]->getTensorDesc().getDims().size() != 4)
            THROW_IE_EXCEPTION << errorPrefix << "must have 4D output tensor. Actual: " << layer->outData[0]->getTensorDesc().getDims().size();

        if (inData->getLayout() != NCHW)
            THROW_IE_EXCEPTION << errorPrefix << "has unsupported layout: " << inData->getLayout();

        const auto precision = inData->getTensorDesc().getPrecision();
        if (_supported_precisions_sizes.find(precision.size()) == _supported_precisions_sizes.end())
            THROW_IE_EXCEPTION << errorPrefix << "has unsupported precision: " << precision.name();

        auto ksizes = layer->GetParamAsUInts("sizes");
        auto strides = layer->GetParamAsUInts("strides");
        auto rates = layer->GetParamAsUInts("rates");
        _auto_pad = layer->GetParamAsString("auto_pad");
        if (!CaselessEq<std::string>()(_auto_pad, "valid")
                && !CaselessEq<std::string>()(_auto_pad, "same_upper")
                && !CaselessEq<std::string>()(_auto_pad, "same_lower"))
            THROW_IE_EXCEPTION <<  errorPrefix << "has unsupported auto_pad value: " << _auto_pad;
        if (ksizes.size() != 2 || strides.size() != 2 || rates.size() != 2)
            THROW_IE_EXCEPTION << errorPrefix << "must have the following attributes with shape {2}: sizes, strides, rates.";

        _ksizes.clear();
        _strides.clear();
        _rates.clear();
        for (size_t i = 0; i < ksizes.size(); i++)
            _ksizes.push_back((int64_t)ksizes[i]);
        for (size_t i = 0; i < strides.size(); i++)
            _strides.push_back((int64_t)strides[i]);
        for (size_t i = 0; i < rates.size(); i++)
            _rates.push_back((int64_t)rates[i]);

        /*** JIT kernel configuration ***/
        jit_eximpat_params jcp;
        SizeVector in_dims = inData->getTensorDesc().getDims();
        jcp.OB = in_dims[0];
        jcp.IC = in_dims[1];
        jcp.IH = in_dims[2];
        jcp.IW = in_dims[3];
        SizeVector out_dims = layer->outData[0]->getTensorDesc().getDims();
        jcp.OH = out_dims[2];
        jcp.OW = out_dims[3];
        jcp.KH = _ksizes[0];
        jcp.KW = _ksizes[1];
        jcp.SH = _strides[0];
        jcp.SW = _strides[1];
        jcp.RH = _rates[0];
        jcp.RW = _rates[1];
        jcp.precision = layer->insData.front().lock()->getPrecision();
        jcp.dtype_size = jcp.precision.size();
        set_pads(_auto_pad, {jcp.IH, jcp.IW, jcp.KH, jcp.KW, jcp.SH, jcp.SW, jcp.RH, jcp.RW});
        jcp.PL = _pads[0];
        jcp.PT = _pads[1];
        jcp.block_size = 1;

        if (mayiuse(x64::avx512_common)) {
            jcp.block_size = cpu_isa_traits<x64::avx512_common>::vlen / jcp.dtype_size;
            eximpat_kernel.reset(new jit_uni_eximpat_kernel_f32<x64::avx512_common>(jcp));
        } else if (mayiuse(x64::avx2)) {
            jcp.block_size = cpu_isa_traits<x64::avx2>::vlen / jcp.dtype_size;
            eximpat_kernel.reset(new jit_uni_eximpat_kernel_f32<x64::avx2>(jcp));
        } else if (mayiuse(x64::sse41)) {
            jcp.block_size = cpu_isa_traits<x64::sse41>::vlen / jcp.dtype_size;
            eximpat_kernel.reset(new jit_uni_eximpat_kernel_f32<x64::sse41>(jcp));
        }
        /*
        if (mayiuse(x64::sse41)) {
            jcp.block_size = cpu_isa_traits<x64::sse41>::vlen / jcp.dtype_size;
            eximpat_kernel.reset(new jit_uni_eximpat_kernel_f32<x64::sse41>(jcp));
        }
         */

        if (eximpat_kernel)
            eximpat_kernel->create_ker();
        /*** JIT kernel configuration finished ***/

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
    } catch (InferenceEngine::details::InferenceEngineException &ex) {
        errorMsg = ex.what();
    }
}

StatusCode ExtractImagePatchesImpl::execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept {
    switch (inputs[0]->getTensorDesc().getPrecision()) {
        case Precision::I8: process_data<PrecisionTrait<Precision::I8>::value_type>(inputs, outputs); break;
        case Precision::I16: process_data<PrecisionTrait<Precision::I16>::value_type>(inputs, outputs); break;
        case Precision::I32: process_data<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs); break;
        case Precision::U8: process_data<PrecisionTrait<Precision::U8>::value_type>(inputs, outputs); break;
        case Precision::U16: process_data<PrecisionTrait<Precision::U16>::value_type>(inputs, outputs); break;
        case Precision::U32: process_data<PrecisionTrait<Precision::U32>::value_type>(inputs, outputs); break;
        case Precision::FP32: process_data<PrecisionTrait<Precision::FP32>::value_type>(inputs, outputs); break;
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

void ExtractImagePatchesImpl::set_pads(const std::string & pad_str, const std::vector<int64_t> & params) {
    const int64_t IH = params[0]; const int64_t IW = params[1];
    const int64_t KH = params[2]; const int64_t KW = params[3];
    const int64_t SH = params[4]; const int64_t SW = params[5];
    const int64_t RH = params[6]; const int64_t RW = params[7];
    const int64_t iwStep = KW + (RW - 1) * (KW - 1);
    const int64_t ihStep = KH + (RH - 1) * (KH - 1);

    int64_t PL = 0, PT = 0;
    if (!CaselessEq<std::string>()(pad_str, "valid")) {
        int64_t PW = (std::ceil(1.f * IW/SW) - 1) * SW + iwStep - IW;
        int64_t PH = (std::ceil(1.f * IH/SH) - 1) * SH + ihStep - IH;

        if ((PW > 0) && (PW < iwStep)) {
            if (PW % 2 == 1) {
                if (CaselessEq<std::string>()(pad_str, "same_lower")) {
                    PL = (PW + 1) / 2;
                } else if (CaselessEq<std::string>()(pad_str, "same_upper")) {
                    PL = (PW - 1) / 2;
                }
            } else {
                PL = PW / 2;
            }
        }
        if ((PH > 0) && (PH < ihStep)) {
            if (PH % 2 == 1) {
                if (CaselessEq<std::string>()(pad_str, "same_lower")) {
                    PT = (PH + 1) / 2;
                } else if (CaselessEq<std::string>()(pad_str, "same_upper")) {
                    PT = (PH - 1) / 2;
                }
            } else {
                PT = PH / 2;
            }
        }
    }
    _pads.clear();
    _pads.push_back(PL);
    _pads.push_back(PT);
}

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
