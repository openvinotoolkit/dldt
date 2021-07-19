// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_kernel.h"

#include <vector>
#include <mkldnn_types.h>
#include <ie_parallel.hpp>
#include <mkldnn_extension_utils.h>
#include "cpu_memcpy.h"
#include "utils/bfloat16.hpp"

#include "cpu/x64/jit_generator.hpp"

using namespace InferenceEngine;
using namespace MKLDNNPlugin;
using namespace mkldnn;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu::x64;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_args_permute, field)

template <cpu_isa_t isa>
struct jit_uni_permute_kernel_f32 : public jit_uni_permute_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_permute_kernel_f32)

    explicit jit_uni_permute_kernel_f32(jit_permute_config_params jcp_) : jit_uni_permute_kernel(jcp_), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);

        loop(jcp.n);

        this->postamble();
    }

    void load(const Xbyak::Xmm &xmm, const Xbyak::Address &addr) {
        switch (jcp.data_size) {
            case 16: movups(xmm, addr); break;
            case 8: movsd(xmm, addr); break;
            case 4: movss(xmm, addr); break;
            case 2: pinsrw(xmm, addr, 0x0); break;
            case 1: pinsrb(xmm, addr, 0x0); break;
        }
    }

    void store(const Xbyak::Address &addr, const Xbyak::Xmm &xmm) {
        switch (jcp.data_size) {
            case 16: movups(addr, xmm); break;
            case 8: movsd(addr, xmm); break;
            case 4: movss(addr, xmm); break;
            case 2: pextrw(addr, xmm, 0x0); break;
            case 1: pextrb(addr, xmm, 0x0); break;
        }
    }

    void loop(int n) {
        mov(reg_work_amount, jcp.dst_block_dims[n]);

        Xbyak::Label main_loop_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label exit_label;

        if (n + 1 == jcp.ndims) {
            if (jcp.src_strides[n] == jcp.dst_strides[n] == 1) {
                uint32_t step = vlen / jcp.data_size;

                L(main_loop_label);
                {
                    cmp(reg_work_amount, step);
                    jl(tail_loop_label, T_NEAR);

                    uni_vmovups(vmm, ptr[reg_src]);
                    uni_vmovups(ptr[reg_dst], vmm);

                    add(reg_src, step * jcp.data_size);
                    add(reg_dst, step * jcp.data_size);
                    sub(reg_work_amount, step);

                    jmp(main_loop_label, T_NEAR);
                }
            }
        }

        L(tail_loop_label); {
            cmp(reg_work_amount, 0);
            je(exit_label, T_NEAR);

            if (n + 1 == jcp.ndims) {
                load(xmm, ptr[reg_src]);
                store(ptr[reg_dst], xmm);
            } else {
                aux_reg_src = reg_src;
                aux_reg_dst = reg_dst;
                push(aux_reg_src);
                push(aux_reg_dst);
                push(reg_work_amount);
                loop(n + 1);
                pop(reg_work_amount);
                pop(reg_dst);
                pop(reg_src);
            }

            add(reg_src, jcp.src_strides[n] * jcp.data_size);
            add(reg_dst, jcp.dst_strides[n] * jcp.data_size);
            sub(reg_work_amount, 1);

            jmp(tail_loop_label, T_NEAR);
        }

        L(exit_label);
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    uint32_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_work_amount = r10;
    Xbyak::Reg64 aux_reg_src = r11;
    Xbyak::Reg64 aux_reg_dst = r12;

    Xbyak::Reg64 reg_params = abi_param1;

    Vmm vmm = Vmm(1);
    Xbyak::Xmm xmm = Xbyak::Xmm(1);
};

PermuteKernel::PermuteKernel(const PermuteParams& params) : params(params) {
    prepareParams();
}

void PermuteKernel::prepareParams() {
    SizeVector src_block_strides(params.src_block_dims.size(), 1);
    SizeVector dst_block_strides(params.dst_block_dims.size(), 1);
    for (int i = params.src_block_dims.size() - 2; i >= 0; i--)
        src_block_strides[i] = src_block_strides[i + 1] * params.src_block_dims[i + 1];
    for (int i = params.dst_block_dims.size() - 2; i >= 0; i--)
        dst_block_strides[i] = dst_block_strides[i + 1] * params.dst_block_dims[i + 1];

    SizeVector tmp_order;
    for (size_t i = 0; i < params.dst_block_order.size(); i++)
        tmp_order.push_back(params.order[params.dst_block_order[i]]);

    const bool byDst = isPermutationsByDstStrides(tmp_order);
    SizeVector first_block_dims     = byDst ? params.dst_block_dims  : params.src_block_dims;
    SizeVector second_block_dims    = byDst ? params.src_block_dims  : params.dst_block_dims;
    SizeVector first_block_order    = byDst ? params.dst_block_order : params.src_block_order;
    SizeVector second_block_order   = byDst ? params.src_block_order : params.dst_block_order;
    SizeVector first_block_strides  = byDst ? dst_block_strides      : src_block_strides;
    SizeVector second_block_strides = byDst ? src_block_strides      : dst_block_strides;

    SizeVector new_first_block_dims = first_block_dims;
    SizeVector new_second_block_dims(second_block_dims.size(), 1);
    SizeVector new_first_block_strides = first_block_strides;
    SizeVector new_second_block_strides(first_block_strides.size(), 1);
    SizeVector new_first_block_order = first_block_order;

    const size_t n_dims = first_block_strides.size();
    if (!byDst) {
        tmp_order.clear();
        for (size_t i = 0; i < first_block_order.size(); i++) {
            const size_t new_value = std::distance(params.order.begin(), std::find(
                                             params.order.begin(), params.order.end(), first_block_order[i]));
            tmp_order.push_back(new_value);
        }
    }

    SizeVector mask(n_dims, 0);
    for (int i = tmp_order.size() - 1; i >= 0; i--) {
        int pos = std::distance(std::find(
                second_block_order.rbegin(), second_block_order.rend(), tmp_order[i]), second_block_order.rend() - 1);
        if (pos != -1) {
            new_second_block_strides[i] = second_block_strides[pos];
            second_block_order.erase(second_block_order.begin() + pos);
            second_block_strides.erase(second_block_strides.begin() + pos);
        } else {
            new_second_block_strides[i] = new_second_block_strides[tmp_order.size() - 1] * first_block_dims[tmp_order.size() - 1];
            mask[i] = 1;
            mask[tmp_order.size() - 1] = 1;
        }
    }
    if (!second_block_order.empty()) {
        int pos = std::distance(tmp_order.begin(), std::find(tmp_order.begin(), tmp_order.end(), second_block_order.front()));

        new_second_block_strides.insert(new_second_block_strides.begin() + pos, second_block_strides.front());
        new_first_block_strides.insert(new_first_block_strides.begin() + pos, new_first_block_strides[pos] * second_block_dims.back());
        new_first_block_order.insert(new_first_block_order.begin() + pos, new_first_block_order[pos]);
        new_first_block_dims.insert(new_first_block_dims.begin() + pos + 1, second_block_dims.back());
        new_first_block_dims[pos] = div_up(new_first_block_dims[pos], new_first_block_dims[pos + 1]);

        mask.insert(mask.begin() + pos + 1, 1);
        mask[pos] = 1;
    }

    SizeVector sorted_first_strides;
    SizeVector sorted_second_strides;
    SizeVector sorted_order;
    SizeVector sorted_main_dims;

    //  support dynamic batch
    const SizeVector order_for_batch = byDst ? params.order : new_first_block_order;
    int batch_ord = std::distance(order_for_batch.begin(), std::find(order_for_batch.begin(), order_for_batch.end(), 0));
    int batch_count = 0;
    int batch_pos = 0;
    for (size_t i = 0; i < new_first_block_order.size(); i++) {
        if (new_first_block_order[i] == batch_ord) {
            batch_count++;
            batch_pos = i;
        }
    }
    if (batch_count == 1) {
        sorted_first_strides.push_back(new_first_block_strides[batch_pos]);
        sorted_second_strides.push_back(new_second_block_strides[batch_pos]);
        sorted_order.push_back(new_first_block_order[batch_pos]);
        sorted_main_dims.push_back(new_first_block_dims[batch_pos]);
        jcp.supported_dynamic_batch = true;
    }

    int n2 = 0;
    for (size_t i = 0; i < mask.size(); i++) {
        if (mask[i] == 0) {
            n2++;
            if (batch_count == 1 && new_first_block_order[i] == batch_ord) {
                continue;
            }
            sorted_first_strides.push_back(new_first_block_strides[i]);
            sorted_second_strides.push_back(new_second_block_strides[i]);
            sorted_order.push_back(new_first_block_order[i]);
            sorted_main_dims.push_back(new_first_block_dims[i]);
        }
    }
    for (size_t i = 0; i < mask.size(); i++) {
        if (mask[i] == 1) {
            sorted_first_strides.push_back(new_first_block_strides[i]);
            sorted_second_strides.push_back(new_second_block_strides[i]);
            sorted_order.push_back(new_first_block_order[i]);
            sorted_main_dims.push_back(new_first_block_dims[i]);
        }
    }

    int max_threads = parallel_get_max_threads();
    const int n_max = 3;    //  max count dims for parallel
    int n = 0;
    int work_amount = sorted_main_dims[0];
    for (size_t i = 1; i < sorted_main_dims.size() && n < n_max; i++) {
        n++;
        if (work_amount >= 4 * max_threads) {   //  4 * max_threads is a specially selected value for best performance
            break;
        }
        work_amount *= sorted_main_dims[i];
    }

    jcp.src_strides = byDst ? sorted_second_strides : sorted_first_strides;
    jcp.dst_strides = byDst ? sorted_first_strides : sorted_second_strides;
    jcp.dst_block_dims = sorted_main_dims;
    jcp.n = std::min(n, n2);
    jcp.ndims = sorted_order.size();
    jcp.data_size = params.data_size;

    if (mayiuse(cpu::x64::avx512_common)) {
        permute_kernel.reset(new jit_uni_permute_kernel_f32<cpu::x64::avx512_common>(jcp));
    } else if (mayiuse(cpu::x64::avx2)) {
        permute_kernel.reset(new jit_uni_permute_kernel_f32<cpu::x64::avx2>(jcp));
    } else if (mayiuse(cpu::x64::sse41)) {
        permute_kernel.reset(new jit_uni_permute_kernel_f32<cpu::x64::sse41>(jcp));
    }

    if (permute_kernel)
        permute_kernel->create_ker();
}

bool PermuteKernel::isPermutationsByDstStrides(const SizeVector& order) const {
    const size_t n_dims = order.size();
    if (n_dims < 4 || std::count(order.begin(), order.end(), 1) != 1)
        return true;

    SizeVector default_order(n_dims);
    std::iota(default_order.begin(), default_order.end(), 0);

    // check order such as 0312, 04123, etc
    SizeVector order_last_to_channel(default_order);
    order_last_to_channel.insert(order_last_to_channel.begin() + 1, order_last_to_channel.back());
    order_last_to_channel.pop_back();
    const size_t dims_before_last = std::accumulate(params.src_block_dims.begin() + 1, params.src_block_dims.end() - 1, 1, std::multiplies<size_t>());
    if (order_last_to_channel == order)
        return dims_before_last <= params.src_block_dims.back();

    // check order such as 0231, 02341, etc
    SizeVector order_channel_to_last(default_order);
    order_channel_to_last.push_back(1);
    order_channel_to_last.erase(order_channel_to_last.begin() + 1);
    const size_t dims_after_first = std::accumulate(params.src_block_dims.begin() + 2, params.src_block_dims.end(), 1, std::multiplies<size_t>());
    if (order_channel_to_last == order)
        return dims_after_first > params.dst_block_dims.back();

    // check order such as 0132
    SizeVector order_spatial_reverse = {0, 1, 3, 2};
    if (order_spatial_reverse == order)
        return params.src_block_dims[n_dims - 2] <= params.src_block_dims[n_dims - 1];

    return true;
}

void PermuteKernel::execute(const uint8_t* src_data, uint8_t* dst_data, const int mb) {
    if (permute_kernel) {
        optimizedExecute(src_data, dst_data, mb);
        return;
    }

    referenceExecute(src_data, dst_data, mb);
}

void PermuteKernel::execute(const uint8_t* src_data, uint8_t* dst_data) {
    const int mb = jcp.dst_block_dims[0];
    if (permute_kernel) {
        optimizedExecute(src_data, dst_data, mb);
        return;
    }

    referenceExecute(src_data, dst_data, mb);
}

void PermuteKernel::optimizedExecute(const uint8_t* src_data, uint8_t* dst_data, const int mb) {
    SizeVector dst_dims = jcp.dst_block_dims;
    const SizeVector dst_strides = jcp.dst_strides;
    const SizeVector src_strides = jcp.src_strides;

    if (dst_dims[0] != mb)
        dst_dims[0] = mb;

    switch (jcp.n) {
        case 1:
            parallel_for(dst_dims[0], [&](int i0) {
                auto arg = jit_args_permute();

                const size_t dst_off = i0 * dst_strides[0];
                const size_t src_off = i0 * src_strides[0];
                arg.src = &src_data[src_off * jcp.data_size];
                arg.dst = &dst_data[dst_off * jcp.data_size];

                (*permute_kernel)(&arg);
            });
            break;
        case 2:
            parallel_for2d(dst_dims[0], dst_dims[1], [&](int i0, int i1) {
                auto arg = jit_args_permute();

                const size_t dst_off = i0 * dst_strides[0] + i1 * dst_strides[1];
                const size_t src_off = i0 * src_strides[0] + i1 * src_strides[1];
                arg.src = &src_data[src_off * jcp.data_size];
                arg.dst = &dst_data[dst_off * jcp.data_size];

                (*permute_kernel)(&arg);
            });
            break;
        case 3:
            parallel_for3d(dst_dims[0], dst_dims[1], dst_dims[2], [&](int i0, int i1, int i2) {
                auto arg = jit_args_permute();

                const size_t dst_off = i0 * dst_strides[0] + i1 * dst_strides[1] + i2 * dst_strides[2];
                const size_t src_off = i0 * src_strides[0] + i1 * src_strides[1] + i2 * src_strides[2];
                arg.src = &src_data[src_off * jcp.data_size];
                arg.dst = &dst_data[dst_off * jcp.data_size];

                (*permute_kernel)(&arg);
            });
            break;
    }
    return;
}

static inline size_t parallel_init(size_t start, size_t nDims, const SizeVector& dims, SizeVector& indexes) {
    for (int j = nDims - 1; j >= 0; j--) {
        indexes[j] = start % dims[j];
        start = start / dims[j];
    }
    return start;
}

static inline void parallel_step(size_t nDims, const SizeVector& dims, SizeVector& indexes) {
    for (int j = nDims - 1; j >= 0; --j) {
        ++indexes[j];
        if (indexes[j] < dims[j])
            break;
        else
            indexes[j] = 0;
    }
}

void PermuteKernel::referenceExecute(const uint8_t* src_data, uint8_t* dst_data, const int mb) {
    SizeVector dst_dims = jcp.dst_block_dims;
    const SizeVector dst_strides = jcp.dst_strides;
    const SizeVector src_strides = jcp.src_strides;
    const size_t data_size = jcp.data_size;
    const size_t ndims = dst_dims.size();

    if (dst_dims[0] != mb)
        dst_dims[0] = mb;

    size_t work_amount = std::accumulate(dst_dims.begin(), dst_dims.end(), 1, std::multiplies<size_t>());

    auto get_idx = [ndims, data_size](const SizeVector& indexes, const SizeVector& strides) {
        size_t idx = 0;
        for (size_t i = 0; i < ndims; ++i)
            idx += indexes[i] * strides[i];
        return idx * data_size;
    };

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector indexes(ndims, 0);
        splitter(work_amount, nthr, ithr, start, end);

        parallel_init(start, ndims, dst_dims, indexes);

        for (size_t iwork = start; iwork < end; ++iwork) {
            const size_t dst_idx = get_idx(indexes, dst_strides);
            const size_t src_idx = get_idx(indexes, src_strides);
            cpu_memcpy(&dst_data[dst_idx], &src_data[src_idx], data_size);

            parallel_step(ndims, dst_dims, indexes);
        }
    });
}
