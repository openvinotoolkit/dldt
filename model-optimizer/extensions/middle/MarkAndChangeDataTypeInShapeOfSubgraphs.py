"""
 Copyright (C) 2018-2021 Intel Corporation

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

from extensions.middle.MarkSubgraphsWithCorrectLayout import MarkSubGraphsWithCorrectLayout
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


class MarkShapeOfSubgraphDataType(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.pass_separator import PostMiddleStart
        return [PostMiddleStart]

    def run_before(self):
        return [MarkSubGraphsWithCorrectLayout]

    def find_and_replace_pattern(self, graph: Graph):
        condition = lambda node: any([out_port.data.get_value() is not None for out_port in node.out_ports().values()])

        start_nodes = []
        shapeof_nodes = set(graph.get_op_nodes(op='ShapeOf'))
        for node in shapeof_nodes:
            start_nodes.extend([n for n in MarkSubGraphsWithCorrectLayout.get_output_nodes(node) if condition(n)])

        nodes_in_shapeof_subgraph = MarkSubGraphsWithCorrectLayout.bfs(start_nodes, shapeof_nodes,
                                                                       condition=condition,
                                                                       direction="bidirectional")
        for node in nodes_in_shapeof_subgraph:
            node['in_shape_subgraph'] = True
            if node.op == 'Const' and node.value.dtype == np.float64:
                node.out_port(0).data.set_value(np.float32(node.out_port(0).data.get_value()))
