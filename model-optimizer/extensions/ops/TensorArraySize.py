# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.common.partial_infer.utils import shape_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class TensorArraySize(Op):
    op = "TensorArraySizeV3"

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'infer': TensorArraySize.array_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def array_infer(node: Node):
        assert len(node.in_nodes()) == 2

        handle = node.in_node(0)
        flow_in = node.in_node(1)

        ta_node = Node(node.graph, str(handle.value))
        assert ta_node.has_valid('size')

        output_value = shape_array(ta_node['size'])
        output_shape = output_value.shape

        for _, out_node in node.graph.out_edges(node.id):
            node.graph.node[out_node]['shape'] = shape_array(output_shape)
            node.graph.node[out_node]['value'] = None if output_value is None else output_value.copy()
