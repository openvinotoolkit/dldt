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

from mo.front.common.partial_infer.utils import mark_input_bins, assign_dims_to_weights, tf_window_op_pad_infer
from mo.front.extractor import spatial_getter
from mo.front.onnx.extractors.utils import get_backend_pad
from mo.graph.graph import Node, Graph
from mo.graph.perm_inputs import PermuteInputs
from mo.ops.op import Op, PermuteAttrs


class Deconvolution(Op):
    op = 'Deconvolution'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',
            'infer': self.infer,
            'in_ports_count': 3,
            'out_ports_count': 1,
        }, attrs)

    def backend_attrs(self):
        return [
            ('dilations', lambda node: ','.join(map(str, node['dilation'][node.spatial_dims]))),
            ('strides', lambda node: ','.join(map(str, node['stride'][node.spatial_dims]))),
            ('pads_begin',
             lambda node: ','.join(map(str, get_backend_pad(node.pad, node.spatial_dims, 0))) if node.has_valid(
                 'pad') else None),
            ('pads_end',
             lambda node: ','.join(map(str, get_backend_pad(node.pad, node.spatial_dims, 1))) if node.has_valid(
                 'pad') else None),
            'auto_pad',
        ]

    @staticmethod
    def infer(node: Node):
        """
        Deconvolution has an input argument that explicitly determines output shape, so in contrast
        to the forward Conv2d we shouldn't infer output shape. We just use this output shape as
        an input shape and pass it to our utilities that computes numeric values for padding.
        They also deliver output shape that is interpreted here as input shape for convolution.
        We need to check that the real input shape and shape inferred by those utility functions match.
        """
        output_shape = np.array(node.in_node(2).value)
        batch = np.array(node.in_node(0).shape)[0]
        output_shape[0] = batch
        kernel_shape = node.in_node(1).shape
        node['kernel_shape'] = kernel_shape
        if output_shape is None or kernel_shape is None or node.spatial_dims is None or node.stride is None:
            return

        if not node.has_valid('kernel_spatial_idx'):
            node['kernel_spatial_idx'] = np.delete([x for x in range(len(kernel_shape))],
                                                   (node.input_feature_channel, node.output_feature_channel))

        if not node.has_valid('dilation'):
            node['dilation'] = np.full([len(output_shape)], 1, dtype=np.int64)

        spatial_dims = node.spatial_dims
        output_spatial = np.array(output_shape[spatial_dims])
        stride_spatial = np.array(node.stride[spatial_dims])
        node['kernel_spatial'] = np.array(kernel_shape[node.kernel_spatial_idx])
        node.pad_spatial_shape, input_spatial_for_check = tf_window_op_pad_infer(
            output_spatial, node.kernel_spatial, stride_spatial, node.auto_pad)

        assert all(input_spatial_for_check == node.in_node(0).shape[spatial_dims])

        pad = np.zeros((len(output_shape), 2), dtype=np.int64)
        pad[spatial_dims] = node.pad_spatial_shape
        node.pad = pad

        node.output = output_shape[node.channel_dims][0]
        node.output_shape = output_shape
        node.out_node().shape = output_shape

        mark_input_bins(node, ['weights'], 1)
        assign_dims_to_weights(node.in_node(1), node.kernel_spatial_idx, node.input_feature_channel,
                               node.output_feature_channel, len(kernel_shape))

        # OK, now we are sure this is a supported Deconvolution layer
        node.type = 'Deconvolution'
        node.op = 'Deconv2D'

        # Add permute_attrs
        PermuteAttrs.create_permute_attrs(node, attrs=[('pad', 'input:0'),
                                                       ('stride', 'input:0'),
                                                       ('output_shape', 'input:0'),
                                                       ('batch_dims', 'input:0'),
                                                       ('channel_dims', 'input:0'),
                                                       ('spatial_dims', 'input:0'),

                                                       ('kernel_shape', 'input:1'),
                                                       ('kernel_spatial_idx', 'input:1'),
                                                       ('input_feature_channel', 'input:1'),
                                                       ('output_feature_channel', 'input:1'),
                                                       ])

        PermuteAttrs.set_permutation(node.in_node(1), node, node.soft_get('get_weights_permute', None))
        PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:1', 'transpose')
        PermuteInputs().set_input_permutation(node.in_node(2), node, 'input:0', 'shape')

        node['force_precision_in_ports'] = {2: 'int64'}
