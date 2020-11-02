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
from typing import Dict

from extensions.ops.Cast import Cast
from mo.front.caffe.extractors.utils import get_canonical_axis_index
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes, Node
from mo.graph.port import Port
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.concat import Concat
from mo.ops.const import Const
from mo.ops.strided_slice import StridedSlice


def create_ss_interval_border(shape, axes, slice_border: Node, port_to_connect: Port):
    mask = np.zeros(len(shape), dtype=np.int64)
    first_part = mask[:axes[0]]
    last_part = mask[axes[-1] + 1:]

    cast = Cast(slice_border.graph, dict(name='Cast', dst_type=np.int64)).create_node()
    cast.in_port(0).connect(port_to_connect)
    concat = create_op_with_const_inputs(slice_border.graph, Concat, port_value_dict={0: first_part, 2: last_part},
                                         op_attrs={'name': 'Concat', 'axis': 0,
                                                   'in_ports_count': 3})
    cast.out_port(0).connect(concat.in_port(1))
    return concat


class ConvertSlice(MiddleReplacementPattern):
    """
    This class converts Slice operation to StridedSlice
    """

    enabled = True
    op = "Slice"
    force_clean_up = True

    def run_after(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('slice', dict(kind='op', op='Slice'))
            ],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: Dict[str, Node]):
        node = match['slice']
        node_name = node.soft_get('name', node.id)

        input_shape = node.in_port(0).data.get_shape()
        axes = node.in_port(3).data.get_value().copy()
        for i, val in enumerate(axes):
            axes[i] = get_canonical_axis_index(input_shape, val)
        axes.sort()
        start_node = node.in_node(1)
        end_node = node.in_node(2)

        ss_begin = create_ss_interval_border(input_shape, axes, start_node, node.in_port(1).get_source())
        ss_end = create_ss_interval_border(input_shape, axes, end_node, node.in_port(2).get_source())
        rename_nodes([(ss_begin, node_name + '/Begin'), (ss_end, node_name + '/End')])

        if node.is_in_port_connected(4):
            steps = node.in_port(4).data.get_value()
        else:
            steps = np.ones([axes.size])

        ss_begin_mask = np.zeros(len(input_shape), dtype=np.int64)
        ss_end_mask = np.zeros(len(input_shape), dtype=np.int64)
        ss_step = np.ones(len(input_shape), dtype=np.int64)

        for i, axis in enumerate(axes):
            ss_begin_mask[axis] = 1
            ss_end_mask[axis] = 1
            ss_step[axis] = steps[i]

        ss_strides = Const(graph, dict(name=node_name + '/Strides', value=ss_step)).create_node()

        ss = StridedSlice(graph, dict(name='ss', new_axis_mask=np.zeros(len(input_shape), dtype=np.int64),
                                      shrink_axis_mask=np.zeros(len(input_shape), dtype=np.int64),
                                      ellipsis_mask=np.zeros(len(input_shape), dtype=np.int64),
                                      begin_mask=ss_begin_mask,
                                      end_mask=ss_end_mask)).create_node()

        node.in_port(0).get_connection().set_destination(ss.in_port(0))
        ss.in_port(1).connect(ss_begin.out_port(0))
        ss.in_port(2).connect(ss_end.out_port(0))
        ss.in_port(3).connect(ss_strides.out_port(0))
        node.out_port(0).get_connection().set_source(ss.out_port(0))

        rename_nodes([(node, node_name + '/ShouldBeDeleted'), (ss, node_name)])
