import pytest

from common.layer_test_class import check_ir_version
from common.tf_layer_test_class import CommonTFLayerTest
from unit_tests.utils.graph import build_graph


class TestReLU6(CommonTFLayerTest):
    def create_relu6_net(self, shape, ir_version):
        """
            Tensorflow net                 IR net

            Input->ReLU6       =>       Input->Clamp

        """

        #
        #   Create Tensorflow model
        #

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            shapes = shape.copy()
            # reshaping
            if len(shapes) >= 3:
                shapes.append(shapes.pop(1))
            input = tf.compat.v1.placeholder(tf.float32, shapes, 'Input')

            tf.nn.relu6(input, name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'ReLU6': {'kind': 'op', 'type': 'Clamp', "max": 6, "min": 0},
                'ReLU6_data': {'shape': shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'ReLU6'),
                                   ('ReLU6', 'ReLU6_data'),
                                   ('ReLU6_data', 'result')
                                   ])

        return tf_net, ref_net

    test_data_precommit = [dict(shape=[1, 3, 50, 100, 224])]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_relu6_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_relu6_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data = [dict(shape=[1]),
                 dict(shape=[1, 224]),
                 pytest.param(dict(shape=[1, 3, 224]), marks=pytest.mark.xfail(reason="*-19053")),
                 dict(shape=[1, 3, 100, 224]),
                 dict(shape=[1, 3, 50, 100, 224])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_relu6(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_relu6_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)