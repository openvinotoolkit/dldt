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

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class Scatter(Op):
    enabled = False

    op = op_type = None
    version = None

    def __init__(self, graph: Graph, attrs: dict):
        assert self.op is not None and self.op_type is not None and self.version is not None, \
            'Please use specialized Scatter operation class, Scatter is base class'

        mandatory_props = {
            'op': self.op,
            'type': self.op_type,
            'version': self.version,

            'is_scatter': True,  # is used for gathering all types of scatters in common transformations
            'infer': self.infer,

            'in_ports_count': 4,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)

        input_shape = node.in_port(0).data.get_shape()
        indices_shape = node.in_port(1).data.get_shape()
        updates_shape = node.in_port(2).data.get_shape()
        assert input_shape is not None and updates_shape is not None and indices_shape is not None, \
            'The node "{}" input shape is None'.format(node_name)

        node.out_port(0).data.set_shape(input_shape)


class ScatterElementsAdd(Scatter):
    op = 'ScatterElementsAdd'
    op_type = None
    version = None


class ScatterElementsDiv(Scatter):
    op = 'ScatterElementsDiv'
    op_type = None
    version = None


class ScatterElementsMax(Scatter):
    op = 'ScatterElementsMax'
    op_type = None
    version = None


class ScatterElementsMin(Scatter):
    op = 'ScatterElementsMin'
    op_type = None
    version = None


class ScatterElementsMul(Scatter):
    op = 'ScatterElementsMul'
    op_type = None
    version = 'opset3'


class ScatterElementsSub(Scatter):
    op = 'ScatterElementsSub'
    op_type = None
    version = None


class ScatterElementsUpdate(Scatter):
    op = op_type = 'ScatterElementsUpdate'
    version = 'opset3'

    @staticmethod
    def infer(node: Node):
        Scatter.infer(node)

        node_name = node.soft_get('name', node.id)
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        assert len(connected_in_ports) == 4, \
            "Incorrect number of inputs for {} node".format(node_name)

        input_value = node.in_port(0).data.get_value()
        input_shape = node.in_port(0).data.get_shape()
        indices_value = node.in_port(1).data.get_value()
        indices_shape = node.in_port(1).data.get_shape()
        updates_value = node.in_port(2).data.get_value()
        updates_shape = node.in_port(2).data.get_shape()
        axis = node.in_port(3).data.get_value()

        data_rank = len(input_shape)
        assert data_rank == len(indices_shape), 'data and indices inputs for node {} must be of the ' \
                                                'same rank. Instead got {} and {}'. \
            format(node.name, data_rank, len(indices_shape))

        assert np.array_equal(indices_shape, updates_shape), 'updates and indices shapes for node {} must be equal. ' \
                                                             'Instead got {} and {}'. \
            format(node.name, indices_shape, updates_shape)

        # compute output value if all inputs are constant
        if input_value is not None and indices_value is not None and updates_value is not None:
            out_value = input_value.copy()
            for idx in np.ndindex(*indices_shape):
                data_idx = list(idx)
                data_idx[axis] = indices_value[idx]
                out_value[tuple(data_idx)] = updates_value[idx]
            node.out_port(0).data.set_value(out_value)


class ScatterAdd(Scatter):
    op = 'ScatterAdd'
    op_type = None
    version = None


class ScatterDiv(Scatter):
    op = 'ScatterDiv'
    op_type = None
    version = None


class ScatterMax(Scatter):
    op = 'ScatterMax'
    op_type = None
    version = None


class ScatterMin(Scatter):
    op = 'ScatterMin'
    op_type = None
    version = None


class ScatterMul(Scatter):
    op = 'ScatterMul'
    op_type = None
    version = None


class ScatterSub(Scatter):
    op = 'ScatterSub'
    op_type = None
    version = None


class ScatterUpdate(Scatter):
    op = op_type = 'ScatterUpdate'
    version = 'opset3'
