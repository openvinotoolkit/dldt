"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np

from mo.front.common.layout import get_features_dim, shape_for_layout
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.ops.unsqueeze import Unsqueeze


class Eltwise1DInputReshape(MiddleReplacementPattern):
    """
    Inserts Reshape before 1-D input to Eltwise if another input of Eltwise is multi-dimensional tensor with the
    same feature size as 1-D input

    Replacer is useful in cases of layout change in MO (for example NHWC-> NCHW translation of TensorFlow models)

    Example:
    Eltwise Mul operation in TF multiplies Tensors by feature dimension with shapes [1,375,500,24] and [24].
    After layout change in MO Eltwise Mul have input shapes [1,24,375,500] and [24]. It is a problem (500!=24).
    We have to insert Reshape layer for Tensor with [24] shape to correspond the feature dimension of
    Tensor [1,24,375,500] shape

    change of graph.graph['layout'] may cause an issue
    change in re-layout function: convert_nhwc_to_nchw(graph) may cause an issue
    """
    enabled = False

    def find_and_replace_pattern(self, graph: Graph):
        layout = graph.graph['layout']
        for eltwise_op_node in graph.get_op_nodes(is_eltwise=True):
                out_shape = eltwise_op_node.out_port().data.get_shape()
                if 4 <= len(out_shape) <= 5:
                    out_features = out_shape[get_features_dim(layout, len(out_shape))]
                    for port, node in eltwise_op_node.in_nodes().items():
                        if len(node.shape) != len(out_shape) and len(node.shape) == 1 and out_features == node.shape[0]:
                            new_shape = shape_for_layout(layout, batch=1, features=out_features, height=1, width=1,
                                                         depth=1 if len(out_shape) == 5 else None)
                            dim_const = Const(graph, {'value': new_shape, 'name': node.id + '/Dim'}).create_node()
                            reshape_op = Reshape(graph, attrs={'dim': new_shape, 'name': node.id + '/Broadcast'}).create_node()

                            eltwise_op_node.in_port(port).get_source().node.out_port(0).get_connection().set_destination(reshape_op.in_port(0))
                            reshape_op.in_port(1).connect(dim_const.out_port(0))

                            reshape_op.out_port(0).connect(eltwise_op_node.in_port(port))


# compute a map of unsqueeze_dims for each producer of eltwise node
def compute_unsqueeze_map_for_eltwise(eltwise_node: Node):
    eltwise_shape = eltwise_node.out_port(0).data.get_shape()
    max_dims = None
    for in_port_idx in eltwise_node.in_ports():
        consumer_port = eltwise_node.in_port(in_port_idx)
        producer_port = consumer_port.get_source()
        producer_shape = producer_port.data.get_shape()
        if max_dims is None or len(producer_shape) > max_dims:
            max_dims = len(producer_shape)

    axis = eltwise_node.axis if eltwise_node.has_valid('axis') else None
    unsqueeze_dims_map = {}
    for in_port_idx in eltwise_node.in_ports():
        consumer_port = eltwise_node.in_port(in_port_idx)
        producer_port = consumer_port.get_source()
        producer_shape = producer_port.data.get_shape()

        unsqueeze_dims = int64_array([])
        # 1. Compute unsqueeze dimensions in the tail
        if len(producer_shape) != max_dims and len(producer_shape) > 0 and axis is not None:
            num_unsqueeze_dims = max_dims - axis - len(producer_shape)
            if num_unsqueeze_dims > 0:
                unsqueeze_dims = np.arange(len(producer_shape), len(producer_shape) + num_unsqueeze_dims,
                                           dtype=np.int64)

        # 2. Compute unsqueeze dimensions in the head
        unsqueeze_dims_head = np.arange(len(eltwise_shape) - len(producer_shape) - len(unsqueeze_dims), dtype=np.int64)

        # Pay attention that unsqueeze dims order makes sense
        # since shape is normalized in the tail first and after in the head
        unsqueeze_dims = np.concatenate((unsqueeze_dims, unsqueeze_dims_head))
        unsqueeze_dims_map[producer_port] = unsqueeze_dims

    return unsqueeze_dims_map


# normalize input shapes for eltwise node by using Unsqueeze operation
def normalize_eltwise_inputs(graph: Graph):
    # Generate a map for producers of eltwise nodes with non-normalized shapes
    # and in this map every producer has another map that reflects normalized shape
    # to a list of eltwise consumers
    mapping = {}
    for eltwise_node in graph.get_op_nodes(is_eltwise=True):
        unsqueeze_dims_map = compute_unsqueeze_map_for_eltwise(eltwise_node)
        for in_port_idx in eltwise_node.in_ports():
            consumer_port = eltwise_node.in_port(in_port_idx)
            producer_port = consumer_port.get_source()
            unsqueeze_dims = unsqueeze_dims_map[producer_port]
            if unsqueeze_dims is not None and len(unsqueeze_dims) > 0:
                unsqueeze_dims = tuple([x for x in unsqueeze_dims])
                if producer_port not in mapping:
                    mapping.update({producer_port: {unsqueeze_dims: [consumer_port]}})
                elif unsqueeze_dims not in mapping[producer_port]:
                    mapping[producer_port].update({unsqueeze_dims: [consumer_port]})
                else:
                    mapping[producer_port][unsqueeze_dims].append(consumer_port)

    # Walk through each produced in the map and insert Unsqueeze nodes between a producer and eltwise nodes
    for producer_port in mapping.keys():
        producer_node = producer_port.node
        for unsqueeze_dims in mapping[producer_port].keys():
            unsqueeze_name = producer_node.soft_get('name', producer_node.id) + '/EltwiseUnsqueeze'
            unsqueeze_node = create_op_with_const_inputs(graph, Unsqueeze, {1: int64_array(list(unsqueeze_dims))},
                                                         {'name': unsqueeze_name})
            unsqueeze_node.in_port(0).connect(producer_port)

            # Insert Unsqueeze with determined unsqueeze dimensions between the current producer and eltwise node
            for consumer_port in mapping[producer_port][unsqueeze_dims]:
                consumer_port.connect(unsqueeze_node.out_port(0))

            # The shape and value adjustments must be explicitly done within the transformation
            # since the transformation is called from Fusing transformation that excludes
            # automatic call of shape inference pass
            producer_port_value = producer_port.data.get_value()
            producer_port_shape = producer_port.data.get_shape()
            new_shape = producer_port_shape.copy()
            for unsqueeze_dim in unsqueeze_dims:
                new_shape = np.insert(new_shape, unsqueeze_dim, 1)
            if producer_port_value is not None:
                unsqueeze_node.out_port(0).data.set_value(np.reshape(producer_port_value, new_shape))
            else:
                unsqueeze_node.out_port(0).data.set_shape(new_shape)
