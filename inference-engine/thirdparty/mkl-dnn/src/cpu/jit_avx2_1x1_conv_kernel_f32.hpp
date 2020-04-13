/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef JIT_AVX2_1x1_CONV_KERNEL_F32_HPP
#define JIT_AVX2_1x1_CONV_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "cpu_memory.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_uni_depthwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx2_1x1_conv_kernel_f32: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_1x1_conv_kernel_f32)

    jit_avx2_1x1_conv_kernel_f32(jit_1x1_conv_conf_t ajcp, jit_conv_conf_t ajcp_dw,
           const primitive_attr_t &attr)
        : jcp(ajcp), jcp_dw(ajcp_dw), attr_(attr)
    {
        this->generate();
        jit_ker = (void (*)(jit_1x1_conv_call_s *))this->getCode();
    }

    ~jit_avx2_1x1_conv_kernel_f32() {
        for (auto inj : eltwise_injectors)
            delete inj;
        eltwise_injectors.clear();

        for (auto inj : depthwise_injectors)
            delete inj;
        depthwise_injectors.clear();

        for (auto inj : quantization_injectors)
            delete inj;
        quantization_injectors.clear();
    }

    static bool post_ops_ok(jit_1x1_conv_conf_t &jcp,
            const primitive_attr_t &attr);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd,
            const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d,
            const primitive_attr_t &attr);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_1x1_conv_conf_t &jcp, const jit_conv_conf_t &jcp_dw = jit_conv_conf_t());

    jit_1x1_conv_conf_t jcp;
    jit_conv_conf_t jcp_dw;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_1x1_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    using ymm_t = const Xbyak::Ymm;

    reg64_t reg_bcast_data = rax;
    reg64_t reg_load_data = rsi;
    reg64_t reg_output_data = rbx;
    reg64_t aux_reg_bcast_data = rdx;
    reg64_t aux1_reg_bcast_data = abi_not_param1;
    reg64_t aux_reg_output_data = rbp;
    reg64_t reg_load_loop_work = r9;
    reg64_t reg_bcast_loop_work = r10;
    reg64_t reg_reduce_loop_work = r11;
    reg64_t load_loop_iter = r13;
    reg64_t aux_reg_load_data = load_loop_iter;
    reg64_t bcast_loop_iter = r14;
    reg64_t reduce_loop_iter = r15;
    reg64_t imm_addr64 = reduce_loop_iter;
    reg64_t reg_reduce_pos_flag = r8;
    reg64_t reg_output_stride = r12;
    reg64_t reg_bias_data = r12;
    reg64_t reg_diff_bias_data = bcast_loop_iter;

    reg64_t reg_oc_off = abi_param1;
    reg64_t reg_d_weights = aux_reg_bcast_data;
    reg64_t reg_d_bias = reduce_loop_iter;

    int reg_diff_bias_data_stack_offt = 0;
    int stack_space_needed = 8;

    ymm_t vreg_bcast = ymm_t(15);
    ymm_t vtmp = ymm_t(14);

    ymm_t ymm_d_weights = Xbyak::Ymm(14);
    ymm_t ymm_d_bias = Xbyak::Ymm(15);

    void generate_bcast_loop(int load_loop_blk);
    void generate_reduce_loop(int load_loop_blk, int ur);
    void generate_diff_bias_loop(int load_loop_blk);

    nstl::vector<jit_uni_eltwise_injector_f32<avx2>*> eltwise_injectors;
    nstl::vector<jit_uni_depthwise_injector_f32<avx2>*> depthwise_injectors;
    nstl::vector<jit_uni_quantization_injector_f32<avx2>*> quantization_injectors;

    void generate();
};

}
}
}

#endif
