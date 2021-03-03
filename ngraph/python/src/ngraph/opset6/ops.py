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
from functools import partial
from typing import Optional

from ngraph.impl import Node
from ngraph.opset_utils import _get_node_factory
from ngraph.utils.decorators import nameable_op
from ngraph.utils.types import (
    NodeInput,
    as_node,
    as_nodes,
)

_get_node_factory_opset6 = partial(_get_node_factory, "opset6")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def ctc_greedy_decoder_seq_len(
        data: NodeInput,
        sequence_length: NodeInput,
        blank_index: Optional[NodeInput] = None,
        merge_repeated: bool = True,
        classes_index_type: str = "i32",
        sequence_length_type: str = "i32",
        name: Optional[str] = None,
) -> Node:
    """Return a node which performs CTCGreedyDecoderSeqLen.

    @param data:            The input 3D tensor. Shape: [batch_size, seq_length, num_classes]
    @param sequence_length: Input 1D tensor with sequence length. Shape: [batch_size]
    @param blank_index:     Scalar or 1D tensor with specifies the class index to use for the blank class.
                            Optional parameter. Default value is num_classes-1.
    @return:                The new node which performs CTCGreedyDecoderSeqLen.
    """
    if blank_index is not None:
        inputs = as_nodes(data, sequence_length, blank_index)
    else:
        inputs = as_nodes(data, sequence_length)

    attributes = {
        "merge_repeated": merge_repeated,
        "classes_index_type": classes_index_type,
        "sequence_length_type": sequence_length_type
    }

    return _get_node_factory_opset6().create("CTCGreedyDecoderSeqLen", inputs, attributes)


@nameable_op
def gather_elements(
    data: NodeInput,
    indices: NodeInput,
    axis: Optional[int] = 0,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs GatherND.

    @param data:       N-D tensor with data for gathering
    @param indices:    N-D tensor with indices by which data is gathered
    @param axis:       axis along which elements are gathered
    @return:           The new node which performs GatherElements
    """
    inputs = as_nodes(data, indices)

    attributes = {
        "axis": axis
    }

    return _get_node_factory_opset6().create("GatherElements", inputs, attributes)


@nameable_op
def mvn(
    data: Node,
    axes: Node,
    normalize_variance: bool,
    eps: float,
    eps_mode: str,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs MeanVarianceNormalization (MVN).

    @param data: The node with data tensor.
    @param axes: The node with axes to reduce on.
    @param normalize_variance: Denotes whether to perform variance normalization.
    @param eps: The number added to the variance to avoid division by zero
               when normalizing the value. Scalar value.
    @param eps_mode: how eps is applied (`inside_sqrt` or `outside_sqrt`)
    @param name: Optional output node name.
    @return The new node performing a MVN operation on input tensor.
    """
    inputs = as_nodes(data, axes)

    attributes = {
        "normalize_variance": normalize_variance,
        "eps": eps,
        "eps_mode": eps_mode
    }

    return _get_node_factory_opset6().create("MVN", inputs, attributes)


@nameable_op
def assign(new_value: NodeInput, variable_id: str, name: Optional[str] = None) -> Node:
    """Return a node which produces the Assign operation.

    @param new_value:    Node producing a value to be assigned to a variable.
    @param variable_id:  Id of a variable to be updated.
    @param name:         Optional name for output node.
    @return Assign node
    """
    return _get_node_factory_opset6().create(
        "Assign",
        [as_node(new_value)],
        {"variable_id": variable_id}
    )


@nameable_op
def read_value(init_value: NodeInput, variable_id: str, name: Optional[str] = None) -> Node:
    """Return a node which produces the Assign operation.

    @param init_value:   Node producing a value to be returned instead of an unassigned variable.
    @param variable_id:  Id of a variable to be read.
    @param name:         Optional name for output node.
    @return ReadValue node
    """
    return _get_node_factory_opset6().create(
        "ReadValue",
        [as_node(init_value)],
        {"variable_id": variable_id}
    )