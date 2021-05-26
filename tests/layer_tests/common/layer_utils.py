import subprocess
import sys

from common.legacy.utils.multiprocessing_utils import multiprocessing_run


def shell(cmd, env=None, cwd=None):
    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        cmd = ['/bin/bash', '-c', "".join(cmd)]
    else:
        cmd = "".join(cmd)
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = p.communicate()
    return p.returncode, stdout, stderr


class BaseInfer:
    def __init__(self, name):
        self.name = name
        self.res = None

    def fw_infer(self, input_data):
        raise RuntimeError("This is base class, please implement infer function for the specific framework")

    def infer(self, input_data):
        self.res = multiprocessing_run(self.fw_infer, [input_data], self.name, timeout=60)
        return self.res


class IEInfer(BaseInfer):
    def __init__(self, model, weights, device):
        super().__init__('Inference Engine')
        self.device = device
        self.model = model
        self.weights = weights

    def fw_infer(self, input_data):
        from openvino.inference_engine import IECore, get_version as ie_get_version

        print("Inference Engine version: {}".format(ie_get_version()))
        print("Creating IE Core Engine...")
        ie = IECore()
        print("Reading network files")
        net = ie.read_network(self.model, self.weights)
        print("Loading network")
        exec_net = ie.load_network(net, self.device)
        print("Starting inference")
        result = exec_net.infer(input_data)

        if "exec_net" in locals():
            del exec_net
        if "ie" in locals():
            del ie

        return result