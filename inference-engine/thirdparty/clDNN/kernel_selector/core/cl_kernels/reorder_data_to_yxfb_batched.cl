// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/reshape_dims.cl"
#include "include/fetch_data.cl"

#include "include/data_types.cl"

///////////////////////// Input Index /////////////////////////
inline uint FUNC(get_input_index)(uint b, uint f, uint y, uint x)
{
#if   INPUT0_SIMPLE
    return GET_DATA_INDEX(INPUT0, b, f, y, x);
#elif defined INPUT0_LAYOUT_BS_F_BSV8__AF8  || \
      defined INPUT0_LAYOUT_BS_F_BSV16__AF8
    return GET_DATA_BS_FYX_BSV8_INDEX(INPUT0, b, f, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_B_FS_YX_FSV16
    return GET_DATA_B_FS_YX_FSV16_INDEX(INPUT0, b, f, y, x);
#elif defined INPUT0_LAYOUT_FS_B_YX_FSV32
    return GET_DATA_FS_B_YX_FSV32_INDEX(INPUT0, b, f, y, x);
#elif defined INPUT0_LAYOUT_B_FS_ZYX_FSV16
    return GET_DATA_B_FS_ZYX_FSV16_INDEX(INPUT0, b, f, 0, y, x);
#elif defined INPUT0_LAYOUT_BS_FS_ZYX_BSV16_FSV16
    return GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX(INPUT0, b, f, 0, y, x);
#elif defined INPUT0_LAYOUT_BS_FS_YX_BSV16_FSV16
    return GET_DATA_BS_FS_YX_BSV16_FSV16_INDEX(INPUT0, b, f, y, x);
#else
#error reorder_data_to_yxfb_batched.cl: input format - not supported
#endif
}

inline void FUNC(get_yxfb_coords_from_linear_idx_no_padding)(uint data_idx, uint* b, uint* f, uint* x, uint* y)
{
    uint tmp_data_idx = data_idx / INPUT0_BATCH_NUM;
    *b = data_idx - tmp_data_idx * INPUT0_BATCH_NUM;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / INPUT0_FEATURE_NUM;
    *f = data_idx - tmp_data_idx * INPUT0_FEATURE_NUM;
    data_idx = tmp_data_idx;

    tmp_data_idx  = data_idx / INPUT0_SIZE_X;
    *x = data_idx - tmp_data_idx * INPUT0_SIZE_X;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / INPUT0_SIZE_Y;
    *y = data_idx - tmp_data_idx * INPUT0_SIZE_Y;
}

__attribute__((intel_reqd_sub_group_size(8)))
KERNEL (reorder_data_to_yxfb_batched)(
    const __global INPUT_REORDER_TYPE* input,
    __global OUTPUT_REORDER_TYPE* output
    #ifdef MEAN_SUBTRACT_IN_BUFFER
    , __global MEAN_SUBTRACT_TYPE* mean_subtract
#endif
    )
{
    uint group_idx = (uint)get_group_id(0) * OUTPUT_BATCH_NUM * 8;

    for(uint i = 0; i < OUTPUT_BATCH_NUM; i++)
    {
        uint output_idx = group_idx + (uint)get_sub_group_local_id();
        if(output_idx >= ELEMENTS_COUNT)
            continue;

        group_idx += 8;

        uint x,y,f,b;
        FUNC_CALL(get_yxfb_coords_from_linear_idx_no_padding)(output_idx, &b,&f,&x,&y);
        const uint input_idx  = FUNC_CALL(get_input_index)(b, f, y, x);

    #if defined MEAN_SUBTRACT_INSIDE_PARAMS
        float res = TO_MEAN_TYPE(input[input_idx]);
        res = MEAN_OP(res, VALUE_TO_SUBTRACT[f % VALUE_TO_SUBTRACT_SIZE]);
    #elif defined MEAN_SUBTRACT_IN_BUFFER
    #if defined MEAN_PER_FEATURE
        MEAN_SUBTRACT_TYPE res = TO_MEAN_TYPE(input[input_idx]);
        res = MEAN_OP(res, mean_subtract[f]);
    #else
        MEAN_SUBTRACT_TYPE res = TO_MEAN_TYPE(input[input_idx]);
        uint8 msv = RESHAPE_DIMS(INPUT0, MEAN_SUBTRACT, b, f, 0, 0, y,x);
        res = MEAN_OP(res, mean_subtract[GET_DATA_INDEX_SAFE(MEAN_SUBTRACT, msv[1], msv[2], msv[5], msv[6])]);
    #endif
    #else
        CALC_TYPE res = TO_CALC_TYPE(input[input_idx]);
    #endif

        output[output_idx] = ACTIVATION_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE_SAT(res), ACTIVATION_PARAMS_TYPED);
    }
}
