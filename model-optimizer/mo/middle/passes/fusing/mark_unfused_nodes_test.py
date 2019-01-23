"""
 Copyright (c) 2018 Intel Corporation

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

from mo.middle.passes.fusing.mark_unfused_nodes import mark_unfused_nodes
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # ScaleShift layer
    'scaleshift_1': {'type': 'ScaleShift', 'kind': 'op', 'op': 'ScaleShift'},
    'scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'scaleshift_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Mul and Add operations
    'mul_1': {'type': 'Mul', 'kind': 'op', 'op': 'Mul'},
    'mul_1_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'mul_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_1': {'type': 'Add', 'kind': 'op', 'op': 'Add'},
    'add_1_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Mul2 and Add2 operations
    'mul_2': {'type': 'Mul', 'kind': 'op', 'op': 'Mul'},
    'mul_2_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'mul_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_2': {'type': 'Add', 'kind': 'op', 'op': 'Add'},
    'add_2_w': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'add_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Concat1 operation
    'concat_1': {'type': 'Concat', 'kind': 'op', 'op': 'Concat'},
    'concat_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Convolutions
    'conv_1': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2D', 'layout': 'NHWC'},
    'conv_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_2': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2D', 'layout': 'NHWC'},
    'conv_2_w': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_2_b': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    # FullyConnected
    'fc_1': {'type': 'FullyConnected', 'kind': 'op', 'op': 'InnerProduct', 'layout': 'NHWC'},
    'fc_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'fc_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'fc_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Placeholders
    'placeholder_2': {'shape': None, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_3': {'shape': None, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_3_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
}


# Unit tests for forward and backward bfs (forward_bfs, backward_bfs)
class MarkFusedNodes(unittest.TestCase):
    def test_mark_unfused_nodes_1(self):
        # Placeholder->ScaleShift->Mul->Add
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('placeholder_1_data', 'concat_1'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([1]), 'value': 6},
                             'add_1_w': {'shape': np.array([1]), 'value': 6},
                             'mul_2_w': {'shape': np.array([1]), 'value': 6},
                             'concat_1_data': {'is_output': True}
                             })

        graph.graph['layout'] = 'NHWC'

        mark_unfused_nodes(graph, '.*mul.*')

        self.assertFalse(graph.node['mul_1']['can_be_fused'], "can_be_fused should be False")
        self.assertFalse(graph.node['mul_2']['can_be_fused'], "can_be_fused should be False")
        self.assertTrue(graph.node['add_1']['can_be_fused'], "can_be_fused should be True")

    def test_mark_unfused_nodes_2(self):
        # Placeholder->ScaleShift->Mul->Add
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('placeholder_1_data', 'concat_1'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([1]), 'value': 6},
                             'add_1_w': {'shape': np.array([1]), 'value': 6},
                             'mul_2_w': {'shape': np.array([1]), 'value': 6},
                             'concat_1_data': {'is_output': True}
                             })
        graph.graph['layout'] = 'NHWC'

        mark_unfused_nodes(graph, '.*')

        self.assertFalse(graph.node['mul_1']['can_be_fused'], "can_be_fused should be False")
        self.assertFalse(graph.node['mul_2']['can_be_fused'], "can_be_fused should be False")
        self.assertFalse(graph.node['add_1']['can_be_fused'], "can_be_fused should be False")
        self.assertFalse(graph.node['placeholder_1']['can_be_fused'], "can_be_fused should be False")
        self.assertFalse(graph.node['concat_1']['can_be_fused'], "can_be_fused should be False")

    def test_mark_unfused_nodes_3(self):
        # Placeholder->ScaleShift->Mul->Add
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('placeholder_1_data', 'concat_1'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([1]), 'value': 6},
                             'add_1_w': {'shape': np.array([1]), 'value': 6},
                             'mul_2_w': {'shape': np.array([1]), 'value': 6},
                             'concat_1_data': {'is_output': True}
                             })
        graph.graph['layout'] = 'NHWC'

        mark_unfused_nodes(graph, 'mul_1,add_1')

        self.assertFalse(graph.node['mul_1']['can_be_fused'], "can_be_fused should be False")
        self.assertFalse(graph.node['add_1']['can_be_fused'], "can_be_fused should be False")
        self.assertTrue(graph.node['mul_2']['can_be_fused'], "can_be_fused should be True")

    def test_mark_unfused_nodes_4(self):
        # Placeholder->ScaleShift->Mul->Add
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('placeholder_1_data', 'concat_1'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'add_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'mul_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'concat_1_data': {'is_output': True}
                             })
        graph.graph['layout'] = 'NHWC'

        mark_unfused_nodes(graph, '')

        self.assertTrue(graph.node['mul_1']['can_be_fused'], "can_be_fused should be True")
        self.assertTrue(graph.node['add_1']['can_be_fused'], "can_be_fused should be True")
        self.assertTrue(graph.node['mul_2']['can_be_fused'], "can_be_fused should be True")

    def test_mark_unfused_nodes_5(self):
        # Placeholder->ScaleShift->Mul->Add
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('placeholder_1_data', 'concat_1'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'add_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'mul_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'concat_1_data': {'is_output': True}
                             })
        graph.graph['layout'] = 'NCHW'

        mark_unfused_nodes(graph, '')

        self.assertTrue(graph.node['mul_1']['can_be_fused'], "can_be_fused should be True")
        self.assertTrue(graph.node['add_1']['can_be_fused'], "can_be_fused should be True")
        self.assertTrue(graph.node['mul_2']['can_be_fused'], "can_be_fused should be True")

        def test_mark_unfused_nodes_5(self):
            # Placeholder->ScaleShift->Mul->Add
            graph = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'add_1'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('add_1_data', 'mul_2'),
                                 ('mul_2_w', 'mul_2'),
                                 ('mul_2', 'mul_2_data'),
                                 ('mul_2_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('placeholder_1_data', 'concat_1'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                                 'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'add_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'concat_1_data': {'is_output': True}
                                 })
            graph.graph['layout'] = 'NCHW'

            mark_unfused_nodes(graph, '')

            self.assertFalse(graph.node['mul_1']['can_be_fused'], "can_be_fused should be False")
            self.assertFalse(graph.node['add_1']['can_be_fused'], "can_be_fused should be False")
            self.assertFalse(graph.node['mul_2']['can_be_fused'], "can_be_fused should be False")

    def test_mark_unfused_nodes_6(self):
        # Placeholder->ScaleShift->Mul->Add
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'mul_2'),
                             ('mul_2_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_2_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('placeholder_1_data', 'concat_1'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_2_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'add_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'mul_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'concat_1_data': {'is_output': True}
                             })
        graph.graph['layout'] = 'NHWC'

        mark_unfused_nodes(graph, '')

        self.assertTrue(graph.node['mul_1']['can_be_fused'], "can_be_fused should be True")
        self.assertTrue(graph.node['add_1']['can_be_fused'], "can_be_fused should be True")
        self.assertTrue(graph.node['mul_2']['can_be_fused'], "can_be_fused should be True")
