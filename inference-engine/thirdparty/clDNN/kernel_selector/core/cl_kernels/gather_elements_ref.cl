// Copyright (c) 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/fetch.cl"
#include "include/include_all.cl"

#define GET_UPDATES_INDEX(prefix, idx_order) CAT(prefix, _GET_INDEX)(idx_order)
#define GET_OUTPUT_INDEX(idx_order) OUTPUT_GET_INDEX(idx_order)

#define IN_ORDER in_b,in_f,in_y,in_x

#define OUT_ORDER out_b,out_f,out_y,out_x
#define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX)(idx_order)

#define INDICES_MAX_DIM 6

KERNEL(gather_nd_ref)(const __global INPUT0_TYPE* data,
                   const __global INPUT1_TYPE* indices,
                   __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif
)
{
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);

    // Calculate indice index
    const uint f = dim2 % OUTPUT_FEATURE_NUM;
    const uint b = dim2 / OUTPUT_FEATURE_NUM;
#if INPUT1_DIMS == 4
    #define IDX_ORDER idx_b,idx_f,idx_y,idx_x
    #define ORDER b,f,y,x
    const uint x = dim0;
    const uint y = dim1;

#elif INPUT1_DIMS == 5
    #define ORDER b,f,z,y,x
    #define IDX_ORDER idx_b,idx_f,idx_z,idx_y,idx_x
    const uint x = dim0;
    const uint y = dim1 % OUTPUT_SIZE_Y;
    const uint z = dim1 / OUTPUT_SIZE_Y;
    // x*y, z

#else
    #define IDX_ORDER idx_b,idx_f,idx_w,idx_z,idx_y,idx_x
    #define ORDER b,f,w,z,y,x
    const uint x = dim0 % OUTPUT_SIZE_X;
    const uint y = dim0 / OUTPUT_SIZE_X;
    const uint z = dim1 % OUTPUT_SIZE_Z;
    const uint w = dim1 / OUTPUT_SIZE_Z;
#endif
    
    // const int out_idx = GET_UPDATES_INDEX(INPUT1, IDX_ORDER);
    const int out_idx = GET_UPDATES_INDEX(INPUT1, ORDER);
    // printf("%d\n", out_idx);
    int axis = AXIS;
    size_t rank = INPUT0_DIMS; // indices_shape.size(), data_shape.size()
    // if (out_idx == 10) {
    //     printf("rank and axis: %d %d\n", rank, axis);
    // }
    // if(out_idx == 10) { printf("Axis: %d\n", axis); }
#if INPUT0_DIMS == 4 
    // size_t data_shape[10] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_X, INPUT0_SIZE_Y, INPUT0_SIZE_Z, INPUT0_SIZE_W};
    size_t data_shape[10] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Y, INPUT0_SIZE_X, INPUT0_SIZE_Z, INPUT0_SIZE_W};
    // size_t indices_shape[10] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_X, INPUT1_SIZE_Y, INPUT1_SIZE_Z, INPUT1_SIZE_W};
    size_t indices_shape[10] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_Y, INPUT1_SIZE_X, INPUT1_SIZE_Z, INPUT1_SIZE_W};
#elif INPUT0_DIMS == 5 
// #else
    size_t data_shape[10] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X, INPUT0_SIZE_W};
    size_t indices_shape[10] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_Z, INPUT1_SIZE_Y, INPUT1_SIZE_X, INPUT1_SIZE_W};
#else
    size_t data_shape[10] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_W, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X};
    size_t indices_shape[10] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_W, INPUT1_SIZE_Z, INPUT1_SIZE_Y, INPUT1_SIZE_X};
#endif

    
    // 6 5 8 1 : b f y x
    // x = 1
    // y = 8
    size_t max_inner_sum = 1, max_outer_sum = 1, outer_sum_inc_data = 1, outer_sum_inc_indices = 1;
    for (size_t i = axis + 1; i < rank; i++)
        max_inner_sum *= indices_shape[i];

    for (int i = 0; i < axis; i++)
        max_outer_sum *= indices_shape[i];

    for (size_t i = axis; i < rank; i++) {
        outer_sum_inc_data *= data_shape[i];
    }
    max_outer_sum *= outer_sum_inc_data;

    for (size_t i = axis; i < rank; i++) { // 2, 3
        outer_sum_inc_indices *= indices_shape[i];
    }

    if(out_idx == 10) {
        // printf("%ld %ld %ld %ld %ld %ld\n", indices[0], indices[1], indices[2], indices[3], indices[4], indices[5]);
        // printf("%ld %ld %ld %ld %ld %ld\n", indices[6], indices[7], indices[8], indices[9], indices[10], indices[11]);
        // printf("%ld %ld %ld %ld %ld %ld\n", indices[12], indices[13], indices[14], indices[15], indices[16], indices[17]);

        printf("aixs: %ld\n", AXIS);
        printf("data: %ld %ld %ld %ld %ld %ld\n", data_shape[0], data_shape[1], data_shape[2], data_shape[3], data_shape[4], data_shape[5]);
        printf("indi: %ld %ld %ld %ld %ld %ld\n", indices_shape[0], indices_shape[1], indices_shape[2], indices_shape[3], indices_shape[4], indices_shape[5]);
    }

//     printf("max_inner_sum: %ld\n", max_inner_sum);
//     printf("outer_sum_inc_data: %ld\n",outer_sum_inc_data);
//     printf("max_inner_sum, max_outer_sum, outer_sum_inc_data: %d %d %d\n",max_inner_sum, max_outer_sum, outer_sum_inc);

// ========================================================================================

    size_t outer_sum = (out_idx / outer_sum_inc_indices);
    outer_sum *= outer_sum_inc_data;
    // size_t outer_sum = (out_idx) * outer_sum_inc_data;
    size_t inner_sum = out_idx % max_inner_sum;
    if (indices[out_idx] < 0 || indices[out_idx] >= data_shape[axis]) {
        printf("indices values of GatherElement exceed data size. %ld %ld \n", out_idx, indices[out_idx]);
        return;
    }
    uint idx = outer_sum + max_inner_sum * indices[out_idx] + inner_sum;
    uint tmp = outer_sum;
    // printf("%d %d, ", out_idx, outer_sum);
    // if(out_idx == 10) { printf("outer_sum: %d\n", tmp); }
    

    INPUT0_TYPE val = data[idx];
    // output[out_idx] = ACTIVATION(val, ACTIVATION_PARAMS);

    // output[out_idx] = TO_OUTPUT_TYPE(axis);
    // output[out_idx] = axis;
// ========================================================================================

    // output[out_idx] = TO_OUTPUT_TYPE(out_idx);

// ========================================================================================
#if HAS_FUSED_OPS
    FUSED_OPS;
    output[out_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
#else
    // output[out_idx] = outer_sum;
    output[out_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
#endif
}

#undef INDICES_MAX_DIM
#undef GET_UPDATES_INDEX
#undef GET_OUTPUT_INDEX
#undef OUT_ORDER
#undef IDX_ORDER
#undef IN_ORDER
