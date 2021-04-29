# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, Node
from mo.ops.op import Op


class MXFFT(Op):
    """
    This operation is intended to read MxNet operations FFT and IFFT.
    The operation MxMetFFT has one attribute: a boolean attribute is_inverse.

    If an operation to read is FFT, then the attribute 'is_inverse' is False, and True otherwise.

    The transformation MXFFTToDFT converts the operation MxNetFFT into MO DFT (if the attribute 'is_inverse'
    is False), or into MO IDFT (otherwise).
    """
    op = 'MXFFT'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'out_ports_count': 1,
            'in_ports_count': 1,
            'infer': self.infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def infer(self, node: Node):
        node_name = node.soft_get('name', node.id)
        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None, 'Input shape of MXFFT node {} must not be None'.format(node_name)
        is_inverse = node.soft_get('is_inverse', False)
        output_shape = input_shape.copy()
        if not is_inverse:
            output_shape[-1] = output_shape[-1] * 2
        else:
            output_shape[-1] = output_shape[-1] // 2
        node.out_port(0).data.set_shape(int64_array(output_shape))
