"""
 Copyright (C) 2020 Intel Corporation

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

import unittest

import numpy as np
from generator import generator, generate

from extensions.ops.MatMul import MatMul, transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

graph_nodes_attrs = {
    'A': {'type': 'Const', 'op': 'Const', 'kind': 'op', 'shape': None, 'value': None},
    'A_data': {'kind': 'data', 'shape': None, 'value': None},
    'B': {'type': 'Const', 'op': 'Const', 'kind': 'op', 'shape': None, 'value': None},
    'B_data': {'kind': 'data', 'shape': None, 'value': None, 'dim_attrs': []},
    'matmul': {'type': 'MatMul', 'op': 'MatMul', 'kind': 'op', 'transpose_a': False, 'transpose_b': False},
    'matmul_data': {'kind': 'data', 'value': None, 'shape': None},
    'output': {'kind': 'op', 'op': 'Result'},
}


graph_edges=[
    ('A', 'A_data'),
    ('B', 'B_data'),
    ('A_data', 'matmul', {'in': 0}),
    ('B_data', 'matmul', {'in': 1}),
    ('matmul', 'matmul_data'),
    ('matmul_data', 'output'),
]


@generator
class TestMatMulValuePropagation(unittest.TestCase):
    @generate(*[
        ([16, 3], np.arange(-5, -5 + 16 * 3).reshape((16, 3)),
         [3, 5], np.arange(0, 3 * 5).reshape((3, 5)),
         False, False),
        ([3, 16], np.arange(-5, -5 + 16 * 3).reshape((3, 16)),
         [3, 5], np.arange(0, 3 * 5).reshape((3, 5)),
         True, False),
        ([5, 8], np.arange(-1, -1 + 5 * 8).reshape((5, 8)),
         [4, 8], np.arange(-2, -2 + 4 * 8).reshape((4, 8)),
         False, True),
        ([8, 8], np.arange(1, 1 + 8 * 8).reshape((8, 8)),
         [4, 8], np.arange(-2, -2 + 4 * 8).reshape((4, 8)),
         True, True),

        ([7, 16, 3], np.arange(0, 0 + 16 * 3 * 7).reshape((7, 16, 3)),
         [3, 5], np.arange(0, 3 * 5).reshape((3, 5)),
         False, False),
        ([1, 3, 16], np.arange(-5, -5 + 16 * 3).reshape((1, 3, 16)),
         [3, 5], np.arange(0, 3 * 5).reshape((3, 5)),
         True, False),
        ([11, 5, 8], np.arange(-1, -1 + 5 * 8 * 11).reshape((11, 5, 8)),
         [11, 4, 8], np.arange(-2, -2 + 4 * 8 * 11).reshape((11, 4, 8)),
         False, True),
        ([1, 3, 5, 8, 8], np.arange(1, 1 + 8 * 8 * 3 * 5).reshape((1, 3, 5, 8, 8)),
         [4, 8], np.arange(-2, -2 + 4 * 8).reshape((4, 8)),
         True, True),
    ])
    def test_value_propagation(self, a_shape, a_value, b_shape, b_value, transpose_a, transpose_b):
        graph = build_graph(
            nodes_attrs=graph_nodes_attrs,
            edges=graph_edges,
            update_attributes={
                'A': {'shape': int64_array(a_shape), 'value': a_value},
                'A_data': {'shape': int64_array(a_shape), 'value': a_value},
                'B': {'shape': int64_array(b_shape), 'value': b_value},
                'B_data': {'shape': int64_array(b_shape), 'value': b_value},
                'matmul': {'transpose_a': transpose_a, 'transpose_b': transpose_b},
                'matmul_data': {'value': None, 'shape': None},
            }
        )
        node = Node(graph, 'matmul')
        MatMul.infer(node)
        node_data = node.out_port(0).get_destination().data.get_value()
        a = a_value
        b = b_value
        if transpose_a:
            a = transpose(a)
        if transpose_b:
            b = transpose(b)
        ref_data = np.matmul(a, b)
        node_data_shape = node_data.shape
        ref_data_shape = ref_data.shape
        msg = "Value propagation for 'matmul' node is not correct."
        self.assertTrue(node_data_shape == ref_data_shape and np.all(node_data == ref_data), msg)
