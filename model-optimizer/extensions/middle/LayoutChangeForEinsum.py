# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.middle.InsertLayoutPropagationTransposes import InsertLayoutPropagationTranspose, \
    is_input_data_in_correct_layout, is_output_data_in_correct_layout
from extensions.ops.einsum import Einsum
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


class LayoutChangeForEinsum(MiddleReplacementPattern):
    """
    The transformation adjusts Einsum equation to NCHW layout.
    Subscripts for tensor of rank greater than three must be adjusted
    to NCHW layout, meaning a label for the last dimension is moved
    to the second position in the subscript.
    There is an exception when the last label in the subscript is ellipsis
    and covers multiple dimensions. In this case subscript is not changed and
    Transpose to get original NHWC layout back is inserted.
    The transformation is only applicable to TensorFlow case.
    """
    enabled = True
    force_shape_inference = True
    graph_condition = [lambda graph: graph.graph['layout'] == 'NHWC']

    def run_after(self):
        return [InsertLayoutPropagationTranspose]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        import extensions.middle.InsertLayoutPropagationTransposes as InsertTransposes
        for einsum in graph.get_op_nodes(type='Einsum'):
            einsum_name = einsum.soft_get('name', einsum.id)
            assert einsum.has_valid('equation'), "Equation attribute is mandatory" \
                                                 " for Einsum node {}".format(einsum_name)
            equation = einsum.equation
            connected_in_ports = [port for port in einsum.in_ports().values() if not port.disconnected()]
            num_inputs = len(connected_in_ports)

            # check if correct_data_layout attribute is set for inputs and output
            input_correct_layout_mask = []
            for input_ind in range(num_inputs):
                input_correct_layout_mask.append(is_input_data_in_correct_layout(einsum, input_ind))
            output_correct_layout_mask = is_output_data_in_correct_layout(einsum, 0)

            # compute a mask of inputs of rank greater than 3 that are required original layout (NCHW)
            # due to presence of ellipsis covering multiple tail dimensions in the corresponding input subscript
            input_ranks = [len(einsum.in_port(port_idx).data.get_shape()) for port_idx in range(num_inputs)]
            output_rank = len(einsum.out_port(0).data.get_shape())
            permuted_equation, is_inputs_adjusted, is_output_adjusted = Einsum.adjust_equation_with_NCHW_layout(
                einsum_name,
                equation,
                input_ranks,
                output_rank, input_correct_layout_mask, output_correct_layout_mask)
            assert len(is_inputs_adjusted) == num_inputs

            # setup adjusted equation
            einsum.equation = permuted_equation

            # insert Transpose node to get NHWC layout back (for inputs) that is required due to specifics of equation
            for input_ind in range(num_inputs):
                if not is_inputs_adjusted[input_ind]:
                    # that means Einsum can only accept input in NHWC layout
                    # so the inserted transpose before the Einsum will convert the layout to NHWC
                    InsertTransposes.insert_transpose(graph, einsum.in_port(input_ind), before_input=True)
            if not is_output_adjusted:
                # that means Einsum can only generate output in NHWC layout
                # so the inserted transpose followed after the output will convert the layout back into NCHW layout
                InsertTransposes.insert_transpose(graph, einsum.out_port(0), before_input=False)
