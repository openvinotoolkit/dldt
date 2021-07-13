#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test conditional compilation.
"""

import glob
import logging
import os
import sys
import numpy as np
import pytest

from proc_utils import cmd_exec  # pylint: disable=import-error
from test_utils import get_lib_sizes, infer_tool, make_build, run_infer  # pylint: disable=import-error


log = logging.getLogger()


@pytest.mark.dependency(name="cc_collect")
def test_cc_collect(model, openvino_ref, test_info,
                    save_session_info, sea_runtool, collector_dir, artifacts):  # pylint: disable=unused-argument
    """Test conditional compilation statistics collection
    :param test_info: custom `test_info` field of built-in `request` pytest fixture.
                      contain a dictionary to store test metadata.
    """
    out = artifacts / artifacts / model[0][0].id / model[0][0].id
    test_info["test_id"] = artifacts / model[0][0].id

    models_list = []
    for model_item in model:
        models_list.append(str(model_item[0].values[0]))

    # cleanup old data if any
    prev_result = glob.glob(f"{out}.pid*.csv")
    for path in prev_result:
        os.remove(path)
    # run use case
    return_code, output = cmd_exec(
        [
            sys.executable,
            str(sea_runtool),
            f"--output={out}",
            f"--bindir={collector_dir}",
            "!",
            sys.executable,
            infer_tool,
            f"-m={models_list}",
            "-d=CPU",
            f"-r={out}"
        ]
    )
    out_csv = glob.glob(f"{out}.pid*.csv")
    test_info["out_csv"] = out_csv

    assert return_code == 0, f"Command exited with non-zero status {return_code}:\n {output}"
    assert len(out_csv) == 1, f'Multiple or none "{out}.pid*.csv" files'


@pytest.mark.dependency(depends=["cc_collect"])
def test_minimized_pkg(model, openvino_root_dir, artifacts):  # pylint: disable=unused-argument
    """Build and install OpenVINO package with collected conditional compilation statistics."""
    out = artifacts / model[0][0].id / model[0][0].id
    install_prefix = out / "install_pkg"
    build_dir = openvino_root_dir / "build_minimized"

    out_csv = glob.glob(f"{out}.pid*.csv")
    assert len(out_csv) == 1, f'Multiple or none "{out}.pid*.csv" files'

    log.info("Building minimized build at %s", build_dir)

    return_code, output = make_build(
        openvino_root_dir,
        build_dir,
        install_prefix,
        cmake_additional_args=[f"-DSELECTIVE_BUILD_STAT={out_csv[0]}"],
        log=log,
    )
    assert return_code == 0, f"Command exited with non-zero status {return_code}:\n {output}"


@pytest.mark.dependency(depends=["cc_collect", "minimized_pkg"])
def test_infer(model, artifacts):
    """Test inference with conditional compiled binaries."""
    out = artifacts / model[0][0].id / model[0][0].id
    minimized_pkg = out / "install_pkg"

    models_list = []
    for model_item in model:
        models_list.append(str(model_item[0].values[0]))

    return_code, output = run_infer(models_list, f"{out}_cc.npz", minimized_pkg)
    assert return_code == 0, f"Command exited with non-zero status {return_code}:\n {output}"


@pytest.mark.dependency(depends=["cc_collect", "minimized_pkg"])
def test_verify(model, openvino_ref, artifacts, tolerance=1e-6):  # pylint: disable=too-many-arguments
    """Test verifying that inference results are equal."""
    out = artifacts / model[0][0].id / model[0][0].id

    models_list = []
    for model_item in model:
        models_list.append(str(model_item[0].values[0]))

    minimized_pkg = out / "install_pkg"
    out_file = f"{out}.npz"
    out_file_cc = f"{out}_cc.npz"
    return_code, output = run_infer(models_list, out_file, openvino_ref)
    assert return_code == 0, f"Command exited with non-zero status {return_code}:\n {output}"
    return_code, output = run_infer(models_list, out_file_cc, minimized_pkg)
    assert return_code == 0, f"Command exited with non-zero status {return_code}:\n {output}"
    reference_results = dict(np.load(out_file, allow_pickle=True))
    inference_results = dict(np.load(out_file_cc, allow_pickle=True))
    for file in reference_results.keys():
        assert (reference_results[file].size ==
                inference_results[file].size
                ), "Reference and inference has different number of model"
        for model_number in range(reference_results[file].size):
            assert sorted(reference_results[file][model_number][0].keys()) == sorted(
                inference_results[file][model_number][0].keys()
            ), "Results have different number of layers"
            for layer in inference_results[file][model_number][0]:
                assert np.allclose(
                    inference_results[file][model_number][0][layer], inference_results[file][model_number][0][layer],
                    tolerance
                ), "Reference and inference results differ"


@pytest.mark.dependency(depends=["cc_collect", "minimized_pkg"])
def test_libs_size(model, openvino_ref, artifacts):  # pylint: disable=unused-argument
    """Test if libraries haven't increased in size after conditional compilation."""
    libraries = ["inference_engine_transformations", "MKLDNNPlugin", "ngraph"]
    minimized_pkg = artifacts / model[0][0].id / model[0][0].id / "install_pkg"
    ref_libs_size = get_lib_sizes(openvino_ref, libraries)
    lib_sizes = get_lib_sizes(minimized_pkg, libraries)

    for lib in libraries:
        lib_size_diff = ref_libs_size[lib] - lib_sizes[lib]
        lib_size_diff_percent = lib_size_diff / ref_libs_size[lib] * 100
        log.info(
            "{}: old - {}kB; new - {}kB; diff = {}kB({:.2f}%)".format(
                lib,
                ref_libs_size[lib] / 1024,
                lib_sizes[lib] / 1024,
                lib_size_diff / 1024,
                lib_size_diff_percent,
            )
        )
    res = [lib for lib in libraries if lib_sizes[lib] > ref_libs_size[lib]]
    assert len(res) == 0, f"These libraries: {res} have increased in size!"
