# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

"""Factory functions for all ngraph ops."""
from typing import Callable, Iterable, List, Optional, Set, Union

import numpy as np
from functools import partial

from ngraph.impl import Node, Shape
from ngraph.impl.op import Constant, Parameter
from ngraph.opset_utils import _get_node_factory
from ngraph.utils.decorators import binary_op, nameable_op, unary_op
from ngraph.utils.input_validation import (
    assert_list_of_ints,
    check_valid_attributes,
    is_non_negative_value,
    is_positive_value,
)
from ngraph.utils.node_factory import NodeFactory
from ngraph.utils.tensor_iterator_types import (
    GraphBody,
    TensorIteratorSliceInputDesc,
    TensorIteratorMergedInputDesc,
    TensorIteratorInvariantInputDesc,
    TensorIteratorBodyOutputDesc,
    TensorIteratorConcatOutputDesc,
)
from ngraph.utils.types import (
    NodeInput,
    NumericData,
    NumericType,
    ScalarData,
    TensorShape,
    as_node,
    as_nodes,
    get_dtype,
    get_element_type,
    get_element_type_str,
    make_constant_node,
)

_get_node_factory_opset5 = partial(_get_node_factory, "opset5")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def batch_norm_inference(
    data: NodeInput,
    gamma: NodeInput,
    beta: NodeInput,
    mean: NodeInput,
    variance: NodeInput,
    epsilon: float,
    name: Optional[str] = None,
) -> Node:
    """Perform layer normalizes a input tensor by mean and variance with appling scale and offset.

    @param data: The input tensor with data for normalization.
    @param gamma: The scalar scaling for normalized value.
    @param beta: The bias added to the scaled normalized value.
    @param mean: The value for mean normalization.
    @param variance: The value for variance normalization.
    @param epsilon: The  number to be added to the variance to avoid division
                    by zero when normalizing a value.
    @param name: The optional name of the output node.
    @return: The new node which performs BatchNormInference.
    """
    inputs = as_nodes(data, gamma, beta, mean, variance)
    return _get_node_factory_opset5().create("BatchNormInference", inputs, {"epsilon": epsilon})


@nameable_op
def gather_nd(
    data: NodeInput,
    indices: NodeInput,
    batch_dims: Optional[int] = 0,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs GatherND.

    @param data:       N-D tensor with data for gathering
    @param indices:    K-D tensor of tuples with indices by which data is gathered
    @param batch_dims: Scalar value of batch dimensions
    @return: The new node which performs GatherND
    """
    inputs = as_nodes(data, indices)

    attributes = {
        "batch_dims": batch_dims
    }

    return _get_node_factory_opset5().create("GatherND", inputs, attributes)


@nameable_op
def log_softmax(data: NodeInput, axis: int, name: Optional[str] = None) -> Node:
    """Apply LogSoftmax operation on each element of input tensor.

    @param data: The tensor providing input data.
    @param axis: An axis along which LogSoftmax should be calculated
    @return: The new node with LogSoftmax operation applied on each element.
    """
    return _get_node_factory_opset5().create("LogSoftmax", [as_node(data)], {"axis": axis})


@nameable_op
def non_max_suppression(
    boxes: NodeInput,
    scores: NodeInput,
    max_output_boxes_per_class: Optional[NodeInput] = None,
    iou_threshold: Optional[NodeInput] = None,
    score_threshold: Optional[NodeInput] = None,
    soft_nms_sigma: Optional[NodeInput] = None,
    box_encoding: str = "corner",
    sort_result_descending: bool = True,
    output_type: str = "i64",
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs NonMaxSuppression.

    @param boxes: Tensor with box coordinates.
    @param scores: Tensor with box scores.
    @param max_output_boxes_per_class: Tensor Specifying maximum number of boxes
                                        to be selected per class.
    @param iou_threshold: Tensor specifying intersection over union threshold
    @param score_threshold: Tensor specifying minimum score to consider box for the processing.
    @param soft_nms_sigma: Tensor specifying the sigma parameter for Soft-NMS.
    @param box_encoding: Format of boxes data encoding.
    @param sort_result_descending: Flag that specifies whenever it is necessary to sort selected
                                   boxes across batches or not.
    @param output_type: Output element type.
    @return: The new node which performs NonMaxSuppression
    """
    if max_output_boxes_per_class is None:
        max_output_boxes_per_class = make_constant_node(0, np.int64)
    if iou_threshold is None:
        iou_threshold = make_constant_node(0, np.float32)
    if score_threshold is None:
        score_threshold = make_constant_node(0, np.float32)
    if soft_nms_sigma is None:
        inputs = as_nodes(
            boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold
        )
    else:
        inputs = as_nodes(
            boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, soft_nms_sigma
        )

    attributes = {
        "box_encoding": box_encoding,
        "sort_result_descending": sort_result_descending,
        "output_type": output_type,
    }

    return _get_node_factory_opset5().create("NonMaxSuppression", inputs, attributes)


@nameable_op
def round(data: NodeInput, mode: str = "half_to_even", name: Optional[str] = None) -> Node:
    """Apply Round operation on each element of input tensor.

    @param data: The tensor providing input data.
    @param mode: Rule to round halfway cases. If set to 'half_to_even' then halfs round to the nearest even
        integer or rounding in such a way that the result heads away from zero if `mode` attribute is
        'half_away_from_zero`.
    @param name: An optional name of the output node.
    @return: The new node with Round operation applied on each element.
    """
    return _get_node_factory_opset5().create("Round", as_nodes(data), {"mode": mode.upper()})


@nameable_op
def lstm_sequence(
        X: NodeInput,
        initial_hidden_state: NodeInput,
        initial_cell_state: NodeInput,
        sequence_lengths: NodeInput,
        W: NodeInput,
        R: NodeInput,
        B: NodeInput,
        hidden_size: int,
        direction: str,
        activations: List[str] = None,
        activations_alpha: List[float] = None,
        activations_beta: List[float] = None,
        clip: float = 0.0,
        name: Optional[str] = None,
) -> Node:
    """Return a node which performs LSTMSequence operation.

    @param X: The input tensor. Shape: [batch_size, seq_length, input_size].
    @param initial_hidden_state:    The hidden state tensor.
                                    Shape: [batch_size, num_directions, hidden_size].
    @param initial_cell_state:      The cell state tensor.
                                    Shape: [batch_size, num_directions, hidden_size].
    @param sequence_lengths:        Specifies real sequence lengths for each batch element.
                                    Shape: [batch_size]. Integer type.
    @param W: Tensor with weights for matrix multiplication operation with input portion of data.
              Expected format: fico
              Shape: [num_directions, 4*hidden_size, input_size].
    @param R: The tensor with weights for matrix multiplication operation with hidden state.
              Expected format: fico
              Shape: [num_directions, 4*hidden_size, hidden_size].
    @param B: The sum of biases (weight and recurrence). Expected format: fico
              Shape: [num_directions, 4*hidden_size].
    @param hidden_size: Specifies hidden state size.
    @param direction: Specifies if the RNN is forward, reverse, or bidirectional.
    @param activations: The list of three activation functions for gates.
    @param activations_alpha: The list of alpha parameters for activation functions.
    @param activations_beta: The list of beta parameters for activation functions.
    @param clip: Specifies bound values [-C, C] for tensor clipping performed before activations.
    @param name: An optional name of the output node.

    @return: The new node represents LSTMSequence. Node outputs count: 3.
    """
    if activations is None:
        activations = ["sigmoid", "tanh", "tanh"]
    if activations_alpha is None:
        activations_alpha = []
    if activations_beta is None:
        activations_beta = []

    node_inputs = as_nodes(X, initial_hidden_state, initial_cell_state, sequence_lengths, W, R, B)

    attributes = {
        "hidden_size": hidden_size,
        "direction": direction.lower(),
        "activations": activations,
        "activations_alpha": activations_alpha,
        "activations_beta": activations_beta,
        "clip": clip,
    }
    return _get_node_factory_opset5().create("LSTMSequence", node_inputs, attributes)


def hsigmoid(data: NodeInput, name: Optional[str] = None,) -> Node:
    """Return a node which performs HSigmoid.

    @param data: Tensor with input data floating point type.
    @return: The new node which performs HSigmoid
    """
    return _get_node_factory_opset5().create("HSigmoid", as_nodes(data), {})


@nameable_op
def gru_sequence(
        X: NodeInput,
        initial_hidden_state: NodeInput,
        sequence_lengths: NodeInput,
        W: NodeInput,
        R: NodeInput,
        B: NodeInput,
        hidden_size: int,
        direction: str,
        activations: List[str] = None,
        activations_alpha: List[float] = None,
        activations_beta: List[float] = None,
        clip: float = 0.0,
        linear_before_reset: bool = False,
        name: Optional[str] = None,
) -> Node:
    """Return a node which performs GRUSequence operation.

    @param X: The input tensor. Shape: [batch_size, seq_length, input_size].
    @param initial_hidden_state:    The hidden state tensor.
                                    Shape: [batch_size, num_directions, hidden_size].
    @param sequence_lengths:        Specifies real sequence lengths for each batch element.
                                    Shape: [batch_size]. Integer type.
    @param W: Tensor with weights for matrix multiplication operation with input portion of data.
              Shape: [num_directions, 3*hidden_size, input_size].
    @param R: The tensor with weights for matrix multiplication operation with hidden state.
              Shape: [num_directions, 3*hidden_size, hidden_size].
    @param B: The sum of biases (weight and recurrence).
              For linear_before_reset set True the shape is [num_directions, 4*hidden_size].
              Otherwise the shape is [num_directions, 3*hidden_size].
    @param hidden_size: Specifies hidden state size.
    @param direction: Specifies if the RNN is forward, reverse, or bidirectional.
    @param activations: The list of three activation functions for gates.
    @param activations_alpha: The list of alpha parameters for activation functions.
    @param activations_beta: The list of beta parameters for activation functions.
    @param clip: Specifies bound values [-C, C] for tensor clipping performed before activations.
    @param linear_before_reset: Flag denotes if the layer behaves according to the modification
                                of GRU described in the formula in the ONNX documentation.
    @param name: An optional name of the output node.

    @return: The new node represents GRUSequence. Node outputs count: 2.
    """
    if activations is None:
        activations = ["sigmoid", "tanh"]
    if activations_alpha is None:
        activations_alpha = []
    if activations_beta is None:
        activations_beta = []

    node_inputs = as_nodes(X, initial_hidden_state, sequence_lengths, W, R, B)

    attributes = {
        "hidden_size": hidden_size,
        "direction": direction.lower(),
        "activations": activations,
        "activations_alpha": activations_alpha,
        "activations_beta": activations_beta,
        "linear_before_reset": linear_before_reset,
        "clip": clip,
    }
    return _get_node_factory_opset5().create("GRUSequence", node_inputs, attributes)


@nameable_op
def rnn_sequence(
        X: NodeInput,
        initial_hidden_state: NodeInput,
        sequence_lengths: NodeInput,
        W: NodeInput,
        R: NodeInput,
        B: NodeInput,
        hidden_size: int,
        direction: str,
        activations: List[str] = None,
        activations_alpha: List[float] = None,
        activations_beta: List[float] = None,
        clip: float = 0.0,
        name: Optional[str] = None,
) -> Node:
    """Return a node which performs RNNSequence operation.

    @param X: The input tensor. Shape: [batch_size, seq_length, input_size].
    @param initial_hidden_state:    The hidden state tensor.
                                    Shape: [batch_size, num_directions, hidden_size].
    @param sequence_lengths:        Specifies real sequence lengths for each batch element.
                                    Shape: [batch_size]. Integer type.
    @param W: Tensor with weights for matrix multiplication operation with input portion of data.
              Shape: [num_directions, hidden_size, input_size].
    @param R: The tensor with weights for matrix multiplication operation with hidden state.
              Shape: [num_directions, hidden_size, hidden_size].
    @param B: The sum of biases (weight and recurrence).
              Shape: [num_directions, hidden_size].
    @param hidden_size: Specifies hidden state size.
    @param direction: Specifies if the RNN is forward, reverse, or bidirectional.
    @param activations: The list of three activation functions for gates.
    @param activations_alpha: The list of alpha parameters for activation functions.
    @param activations_beta: The list of beta parameters for activation functions.
    @param clip: Specifies bound values [-C, C] for tensor clipping performed before activations.
    @param name: An optional name of the output node.

    @return: The new node represents RNNSequence. Node outputs count: 2.
    """
    if activations is None:
        activations = ["tanh"]
    if activations_alpha is None:
        activations_alpha = []
    if activations_beta is None:
        activations_beta = []

    inputs = as_nodes(X, initial_hidden_state, sequence_lengths, W, R, B)

    attributes = {
        "hidden_size": hidden_size,
        "direction": direction.lower(),
        "activations": activations,
        "activations_alpha": activations_alpha,
        "activations_beta": activations_beta,
        "clip": clip,
    }

    return _get_node_factory_opset5().create("RNNSequence", inputs, attributes)


@nameable_op
def loop(
    trip_count: NodeInput,
    execution_condition: NodeInput,
    inputs: List[Node],
    graph_body: GraphBody,
    slice_input_desc: List[TensorIteratorSliceInputDesc],
    merged_input_desc: List[TensorIteratorMergedInputDesc],
    invariant_input_desc: List[TensorIteratorInvariantInputDesc],
    body_output_desc: List[TensorIteratorBodyOutputDesc],
    concat_output_desc: List[TensorIteratorConcatOutputDesc],
    body_condition_output_idx: int,
    current_iteration_input_idx: int = -1,
    name: Optional[str] = None,
) -> Node:
    """Perform recurrent execution of the network described in the body, iterating through the data.

    @param trip_count: A scalar or 1D tensor with 1 element specifying
        maximum number of iterations.
    @param execution_condition: A scalar or 1D tensor with 1 element
        specifying whether to execute the first iteration or not.
    @param      inputs:                The provided to TensorIterator operator.
    @param      graph_body:            The graph representing the body we execute.
    @param      slice_input_desc:      The descriptors describing sliced inputs, that is nodes
                                       representing tensors we iterate through, processing single
                                       data slice in one iteration.
    @param      merged_input_desc:     The descriptors describing merged inputs, that is nodes
                                       representing variables with initial value at first iteration,
                                       which may be changing through iterations.
    @param      invariant_input_desc:  The descriptors describing invariant inputs, that is nodes
                                       representing variable with persistent value through all
                                       iterations.
    @param      body_output_desc:      The descriptors describing body outputs from specified
                                       iteration.
    @param      concat_output_desc:    The descriptors describing specified output values through
                                       all the iterations concatenated into one node.
    @param      body_condition_output_idx:    TODO: add desc
    @param      current_iteration_input_idx:  TODO: add desc
    @return: The new node which performs Loop.
    """
    attributes = {
        "body": graph_body.serialize(),
        "input_descriptions": {"slice_input_desc": [desc.serialize() for desc in slice_input_desc],
                               "merged_input_desc": [desc.serialize() for desc in merged_input_desc],
                               "invariant_input_desc": [desc.serialize() for desc in invariant_input_desc]},
        "output_descriptions": {"body_output_desc": [desc.serialize() for desc in body_output_desc],
                                "concat_output_desc": [desc.serialize() for desc in concat_output_desc]},
        "special_body_ports": {"body_condition_output_idx": body_condition_output_idx,
                               "current_iteration_input_idx": current_iteration_input_idx}
    }
    return _get_node_factory_opset5().create("Loop", as_nodes(trip_count, execution_condition, *inputs),
                                             attributes)
