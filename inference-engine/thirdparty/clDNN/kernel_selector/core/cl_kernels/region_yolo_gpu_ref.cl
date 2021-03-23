// Copyright (c) 2017 Intel Corporation
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

inline INPUT0_TYPE FUNC(logistic_activate)(INPUT0_TYPE x) {
    return 1. / (1. + exp(-x));
}

inline int FUNC(output_index)(int batch, int region_num, int x, int y, int xy, int feature_offset) {
#if DO_SOFTMAX
    return OUTPUT_GET_INDEX(batch, feature_offset * INPUT0_SIZE_X * INPUT0_SIZE_Y + xy, 1, 1);
#else
    return OUTPUT_GET_INDEX(batch, feature_offset, y, x);
#endif
}

KERNEL (region_yolo_ref)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
    int xy = get_global_id(0);
    int region_num = get_global_id(1);
    int batch = get_global_id(2);
    int x_index = xy % INPUT0_SIZE_X;
    int y_index = (xy / INPUT0_SIZE_X) % (INPUT0_SIZE_Y);

    /// [x, y, width, height, objectness score, class score]
    /// x,y
    int region_offset = region_num * (COORDS + CLASSES + 1);
    int in_i = INPUT0_GET_INDEX(batch, 0 + region_offset, y_index, x_index);
    int out_i = FUNC_CALL(output_index)(batch, region_num, x_index, y_index, xy, 0 + region_offset);
    output[out_i] = FUNC_CALL(logistic_activate)(input[in_i]);

    in_i = INPUT0_GET_INDEX(batch, 1 + region_offset, y_index, x_index);
    out_i = FUNC_CALL(output_index)(batch, region_num, x_index, y_index, xy, 1 + region_offset);
    output[out_i] = FUNC_CALL(logistic_activate)(input[in_i]);

    /// width,height
    in_i = INPUT0_GET_INDEX(batch, 2 + region_offset, y_index, x_index);
    out_i = FUNC_CALL(output_index)(batch, region_num, x_index, y_index, xy, 2 + region_offset);
    output[out_i] = input[in_i];

    in_i = INPUT0_GET_INDEX(batch, 3 + region_offset, y_index, x_index);
    out_i = FUNC_CALL(output_index)(batch, region_num, x_index, y_index, xy, 3 + region_offset);
    output[out_i] = input[in_i];

    /// objectness score
    in_i = INPUT0_GET_INDEX(batch, COORDS + region_offset, y_index, x_index);
    out_i = FUNC_CALL(output_index)(batch, region_num, x_index, y_index, xy, COORDS + region_offset);
    output[out_i] = FUNC_CALL(logistic_activate)(input[in_i]);

    /// class score(confidence)
#if DO_SOFTMAX
    in_i = INPUT0_GET_INDEX(batch, COORDS + 1 + region_offset, y_index, x_index);
    INPUT0_TYPE max_value = input[in_i];
    for (int j = 1; j < CLASSES; j++) {
        in_i = INPUT0_GET_INDEX(batch, COORDS + 1 + j + region_offset, y_index, x_index);
        max_value = max(max_value, input[in_i]);
    }

    OUTPUT_TYPE expSum = 0;
    for (int j = 0; j < CLASSES; j++) {
        in_i = INPUT0_GET_INDEX(batch, COORDS + 1 + j + region_offset, y_index, x_index);
        out_i = FUNC_CALL(output_index)(batch, region_num, x_index, y_index, xy, COORDS + 1 + j + region_offset);
        output[out_i] = exp(input[in_i] - max_value);
        expSum += output[out_i];
    }

    for (int j = 0; j < CLASSES; j++) {
        out_i = FUNC_CALL(output_index)(batch, region_num, x_index, y_index, xy, COORDS + 1 + j + region_offset);
        output[out_i] /= expSum;
    }
#else
    for (int j = 0; j < CLASSES; j++)
    {
        in_i = INPUT0_GET_INDEX(batch, COORDS + 1 + j + region_offset, y_index, x_index);
        output[in_i] = FUNC_CALL(logistic_activate)(input[in_i]);
    }
#endif
}
