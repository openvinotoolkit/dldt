# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml or .onnx file with a trained model.", required=True, type=str)
    args.add_argument("-i", "--input", help="Required. Path to an image files", required=True,
                      type=str, nargs="+")
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "Absolute MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the "
                           "kernels implementations", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. Sample "
                           "will look for a suitable plugin for device specified. Default value is CPU", default="CPU",
                      type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)
    args.add_argument("--mean_val_r", "-mean_val_r",
                      help="Optional. Mean value of red channel for mean value subtraction in postprocessing ", default=0,
                      type=float)
    args.add_argument("--mean_val_g", "-mean_val_g",
                      help="Optional. Mean value of green channel for mean value subtraction in postprocessing ", default=0,
                      type=float)
    args.add_argument("--mean_val_b", "-mean_val_b",
                      help="Optional. Mean value of blue channel for mean value subtraction in postprocessing ", default=0,
                      type=float)
    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")

    # Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
    model = args.model
    log.info(f"Loading network:\n\t{model}")
    net = ie.read_network(model=model)

    assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    net.batch_size = len(args.input)

    # Read and pre-process input images
    n, c, h, w = net.input_info[input_blob].input_data.shape
    images = np.ndarray(shape=(n, c, h, w))
    for i in range(n):
        image = cv2.imread(args.input[i])
        if image.shape[:-1] != (h, w):
            log.warning(f"Image {args.input[i]} is resized from {image.shape[:-1]} to {(h, w)}")
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[i] = image
    log.info(f"Batch size is {n}")

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    # Start sync inference
    log.info("Starting inference")
    res = exec_net.infer(inputs={input_blob: images})

    # Processing output blob
    log.info("Processing output blob")
    res = res[out_blob]
    # Post process output
    for batch, data in enumerate(res):
        # Clip values to [0, 255] range
        data = np.swapaxes(data, 0, 2)
        data = np.swapaxes(data, 0, 1)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data[data < 0] = 0
        data[data > 255] = 255
        data = data[::] - (args.mean_val_r, args.mean_val_g, args.mean_val_b)
        out_img = os.path.join(os.path.dirname(__file__), f"out_{batch}.bmp")
        cv2.imwrite(out_img, data)
        log.info(f"Result image was saved to {out_img}")
    log.info("This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n")


if __name__ == '__main__':
    sys.exit(main() or 0)
