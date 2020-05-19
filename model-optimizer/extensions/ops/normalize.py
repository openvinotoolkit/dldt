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

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.front.common.partial_infer.utils import mark_input_bins
from mo.graph.graph import Graph, Node
from mo.ops.op import Op
from mo.utils.utils import convert_param_type


class NormalizeOp(Op):
    op = 'Normalize'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'eps': None,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': __class__.infer
        }, attrs)

        if 'across_spatial' in self.attrs and isinstance(self.attrs['across_spatial'], str):
            self.attrs['across_spatial'] = int(self.attrs['across_spatial'])

        if 'channel_shared' in self.attrs and isinstance(self.attrs['channel_shared'], str):
            self.attrs['channel_shared'] = int(self.attrs['channel_shared'])

        self.attrs['across_spatial'] = bool(self.attrs['across_spatial'])
        self.attrs['channel_shared'] = bool(self.attrs['channel_shared'])

    def supported_attrs(self):
        return ['eps', 'eps_mode',
                ('across_spatial', lambda node: convert_param_type(node, 'across_spatial', bool, int)),
                ('channel_shared', lambda node: convert_param_type(node, 'channel_shared', bool, int)),
                ]

    @staticmethod
    def infer(node: Node):
        mark_input_bins(node)
        copy_shape_infer(node)
