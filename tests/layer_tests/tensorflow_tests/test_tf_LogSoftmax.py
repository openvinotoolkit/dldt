# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from distutils.version import LooseVersion

import numpy as np
import pytest
from common.layer_test_class import check_ir_version
from common.tf_layer_test_class import CommonTFLayerTest
from mo.front.common.partial_infer.utils import int64_array
from unit_tests.utils.graph import build_graph


class TestLogSoftmax(CommonTFLayerTest):
    disable_input_layout_conversion = True

    def create_log_softmax_net(self, shape, reduction_axis, ir_version):
        """
            Tensorflow net                 IR net

            Input->LogSoftmax       =>       Input->Softmax->Log

        """
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.float32, shape, 'Input')
            if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
                tf.nn.log_softmax(input, name='Operation', axis=reduction_axis)
            else:
                tf.nn.log_softmax(input, axis=reduction_axis, name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        reduce_sum_shape = np.copy(shape)
        reduce_sum_shape[reduction_axis] = 1

        if check_ir_version(10, None, ir_version):
            ref_nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter', 'shape': shape},
                'input_data': {'shape': shape, 'kind': 'data', 'value': None},
                'reduce_max_axis_val': {'shape': int64_array([reduction_axis]).shape,
                                        'kind': 'data',
                                        'value': int64_array([reduction_axis])},
                'reduce_max_axis': {'type': 'Const', 'kind': 'op', 'shape': 1},
                'reduce_max_axis_data': {'shape': int64_array([1]), 'kind': 'data', 'value': None},
                'reduce_max': {'type': 'ReduceMax', 'kind': 'op', 'keep_dims': True},
                'reduce_max_data': {'shape': reduce_sum_shape, 'kind': 'data', 'value': None},
                'sub_first': {'type': 'Subtract', 'kind': 'op'},
                'sub_first_data': {'shape': shape, 'kind': 'data', 'value': None},
                'reduce_sum_axis_val': {'shape': int64_array([reduction_axis]).shape,
                                        'kind': 'data',
                                        'value': int64_array([reduction_axis])},
                'reduce_sum_axis': {'type': 'Const', 'kind': 'op', 'shape': 1},
                'reduce_sum_axis_data': {'shape': int64_array([1]), 'kind': 'data', 'value': None},
                'reduce_sum': {'type': 'ReduceSum', 'kind': 'op', 'keep_dims': True},
                'reduce_sum_data': {'shape': reduce_sum_shape, 'kind': 'data', 'value': None},
                'exp': {'type': 'Exp', 'kind': 'op'},
                'exp_data': {'shape': shape, 'kind': 'data', 'value': None},
                'log': {'type': 'Log', 'kind': 'op'},
                'log_data': {'shape': reduce_sum_shape, 'kind': 'data', 'value': None},
                'sub_second': {'type': 'Subtract', 'kind': 'op'},
                'sub_second_data': {'shape': shape, 'kind': 'data', 'value': None},
                'result': {'kind': 'op', 'type': 'Result'},
            }

            ref_edges = [
                ('input', 'input_data'),
                ('reduce_max_axis_val', 'reduce_max_axis'),
                ('reduce_max_axis', 'reduce_max_axis_data'),
                ('reduce_max_axis_data', 'reduce_max', {'in': 1}),
                ('reduce_max', 'reduce_max_data'),
                ('input_data', 'reduce_max', {'out': 0, 'in': 0}),
                ('input_data', 'sub_first', {'out': 0, 'in': 0}),
                ('reduce_max_data', 'sub_first', {'in': 1}),
                ('sub_first', 'sub_first_data'),
                ('reduce_sum_axis_val', 'reduce_sum_axis'),
                ('reduce_sum_axis', 'reduce_sum_axis_data'),
                ('reduce_sum_axis_data', 'reduce_sum', {'in': 1}),
                ('reduce_sum', 'reduce_sum_data'),
                ('sub_first_data', 'exp'),
                ('exp', 'exp_data'),
                ('exp_data', 'reduce_sum', {'in': 0}),
                ('reduce_sum_data', 'log'),
                ('log', 'log_data'),
                ('log_data', 'sub_second', {'in': 1}),
                ('sub_second', 'sub_second_data'),
                ('sub_first_data', 'sub_second', {'out': 0, 'in': 0}),
                ('sub_second_data', 'result'),
            ]

            ref_net = build_graph(ref_nodes_attributes, ref_edges)

        # TODO ref graph is incorrect
        ref_net = None
        return tf_net, ref_net

    test_data_precommit = [
        dict(shape=[3, 2, 3, 7, 6], reduction_axis=-1),
    ]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_log_softmax_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_log_softmax_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   kwargs_to_prepare_input={'min_value': 1, 'max_value': 255})

    test_data = [dict(shape=[1], reduction_axis=-1),
                 dict(shape=[2, 5], reduction_axis=-1),
                 dict(shape=[5, 3, 7, 4], reduction_axis=-1),
                 dict(shape=[3, 2, 3, 7, 6], reduction_axis=-1)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_log_softmax(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_log_softmax_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   kwargs_to_prepare_input={'min_value': 1, 'max_value': 255})
