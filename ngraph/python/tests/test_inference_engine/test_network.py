# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

from openvino.inference_engine import IECore, IENetwork, ExecutableNetwork, DataPtr, InputInfoPtr, InputInfoCPtr

import os


def model_path(is_myriad=False):
    path_to_repo = os.environ["MODELS_PATH"]
    if not is_myriad:
        test_xml = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp32.xml')
        test_bin = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp32.bin')
    else:
        test_xml = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp16.xml')
        test_bin = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp16.bin')
    return (test_xml, test_bin)

test_net_xml, test_net_bin = model_path()

def test_name():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert not(isinstance(net.input_info['data'], InputInfoCPtr))
    assert isinstance(net.input_info['data'], InputInfoPtr)
    assert net.name == "test_model"


def test_batch_size_getter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.batch_size == 1


def test_batch_size_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.batch_size = 4
    assert net.batch_size == 4
    assert net.input_info['data'].input_data.shape == [4, 3, 32, 32]


def test_reshape():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.reshape({"data": (2, 3, 32, 32)})


def test_batch_size_after_reshape():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.reshape({'data': [4, 3, 32, 32]})
    assert net.batch_size == 4
    assert net.input_info['data'].input_data.shape == [4, 3, 32, 32]
    net.reshape({'data': [8, 3, 32, 32]})
    assert net.batch_size == 8
    assert net.input_info['data'].input_data.shape == [8, 3, 32, 32]

def test_outputs():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert isinstance(net.outputs['fc_out'], DataPtr)
    assert net.outputs['fc_out'].layout == "NC"
    assert net.outputs['fc_out'].precision == "FP32"
    assert net.outputs['fc_out'].shape == [1, 10]


def test_output_precision_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.outputs['fc_out'].precision == "FP32"
    net.outputs['fc_out'].precision = "I8"
    assert net.outputs['fc_out'].precision == "I8"


def test_add_outputs():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.add_outputs('28/Reshape')
    net.add_outputs(['29/WithoutBiases'])
    assert sorted(net.outputs) == ['28/Reshape', '29/WithoutBiases', 'fc_out']


def test_add_outputs_with_port():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.add_outputs(('28/Reshape', 0))
    net.add_outputs([('29/WithoutBiases', 0)])
    assert sorted(net.outputs) == ['28/Reshape', '29/WithoutBiases', 'fc_out']


def test_add_outputs_with_and_without_port():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.add_outputs('28/Reshape')
    net.add_outputs([('29/WithoutBiases', 0)])
    assert sorted(net.outputs) == ['28/Reshape', '29/WithoutBiases', 'fc_out']


def test_multi_out_data():
    # Regression test CVS-23965
    # Check that DataPtr for all output layers not copied between outputs map  items
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.add_outputs(['28/Reshape'])
    assert "28/Reshape" in net.outputs and "fc_out" in net.outputs
    assert isinstance(net.outputs["28/Reshape"], DataPtr)
    assert isinstance(net.outputs["fc_out"], DataPtr)
    assert net.outputs["28/Reshape"].name == "28/Reshape" and net.outputs["28/Reshape"].shape == [1, 5184]
    assert net.outputs["fc_out"].name == "fc_out" and net.outputs["fc_out"].shape == [1, 10]
    pass
