# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.utils import shape_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class TensorArrayReader(Op):
    op = "TensorArrayReadV3"

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'infer': TensorArrayReader.array_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def array_infer(node: Node):
        assert len(node.in_nodes()) == 3

        handle = node.in_node(0)

        ta_node = Node(node.graph, str(handle.value))
        assert ta_node.has_valid('element_shape')

        data_shape = ta_node['element_shape']

        output_shape = data_shape

        for _, out_node in node.graph.out_edges(node.id):
            node.graph.node[out_node]['shape'] = shape_array(output_shape)
            node.graph.node[out_node]['value'] = None
