#!/usr/bin/env bash

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

printf "\e[0;33mWARNING: If you get an error when running the demo in the Docker container, you may need to install additional packages. To do this, run the container as root (-u 0) and run install_openvino_dependencies.sh script. If you get a package-independent error, try setting additional parameters using -sample-options.\e[0m\n"

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]-$0}" )" && pwd )"

. "$ROOT_DIR/utils.sh"

usage() {
    echo "Classification demo using public SqueezeNet topology"
    echo "-d name     specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD are acceptable. Sample will look for a suitable plugin for device specified"
    echo "-help            print help message"
    echo "-sample-options   specify command line arguments for the sample"
    exit 1
}

trap 'error ${LINENO}' ERR

target="CPU"

# parse command line options
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -h | -help | --help)
    usage
    ;;
    -d)
    target="$2"
    echo target = "${target}"
    shift
    ;;
    -sample-options)
    sampleoptions="$2 $3 $4 $5 $6"
    echo sample-options = "${sampleoptions}"
    shift
    ;;
    *)
    # unknown option
    ;;
esac
shift
done

target_precision="FP16"

printf "target_precision = ${target_precision}\n"

models_path="$HOME/openvino_models/models"
models_cache="$HOME/openvino_models/cache"
irs_path="$HOME/openvino_models/ir"

model_name="squeezenet1.1"

target_image_path="$ROOT_DIR/car.png"

run_again="Then run the script again\n\n"
dashes="\n\n###################################################\n\n"


if [ -e "$ROOT_DIR/../../bin/setupvars.sh" ]; then
    setupvars_path="$ROOT_DIR/../../bin/setupvars.sh"
else
    printf "Error: setupvars.sh is not found\n"
fi

if ! . "$setupvars_path" ; then
    printf "Unable to run ./setupvars.sh. Please check its presence. ${run_again}"
    exit 1
fi

# Step 1. Download the Caffe model and the prototxt of the model
printf "${dashes}"
printf "\n\nDownloading the Caffe model and the prototxt"

cur_path=$PWD

printf "\nInstalling dependencies\n"

if [[ -f /etc/centos-release ]]; then
    DISTRO="centos"
elif [[ -f /etc/lsb-release ]]; then
    DISTRO="ubuntu"
fi

if [[ $DISTRO == "centos" ]]; then
    # check installed Python version
    if command -v python3.5 >/dev/null 2>&1; then
        python_binary=python3.5
        pip_binary=pip3.5
    fi
    if command -v python3.6 >/dev/null 2>&1; then
        python_binary=python3.6
        pip_binary=pip3.6
    fi
elif [[ $DISTRO == "ubuntu" ]]; then
    python_binary=python3
    pip_binary=pip3
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # check installed Python version
    if command -v python3.7 >/dev/null 2>&1; then
        python_binary=python3.7
        pip_binary=pip3.7
    elif command -v python3.6 >/dev/null 2>&1; then
        python_binary=python3.6
        pip_binary=pip3.6
    elif command -v python3.5 >/dev/null 2>&1; then
        python_binary=python3.5
        pip_binary=pip3.5
    else
        python_binary=python3
        pip_binary=pip3
    fi
fi

if ! command -v $python_binary &>/dev/null; then
    printf "\n\nPython 3.5 (x64) or higher is not installed. It is required to run Model Optimizer, please install it. ${run_again}"
    exit 1
fi

"$pip_binary" install -r "$ROOT_DIR/../open_model_zoo/tools/downloader/requirements.in"

downloader_dir="${INTEL_OPENVINO_DIR}/deployment_tools/open_model_zoo/tools/downloader"

model_dir=$("$python_binary" "$downloader_dir/info_dumper.py" --name "$model_name" |
    "$python_binary" -c 'import sys, json; print(json.load(sys.stdin)[0]["subdirectory"])')

downloader_path="$downloader_dir/downloader.py"

print_and_run "$python_binary" "$downloader_path" --name "$model_name" --output_dir "${models_path}" --cache_dir "${models_cache}"

ir_dir="${irs_path}/${model_dir}/${target_precision}"

if [ ! -e "$ir_dir" ]; then
    # Step 2. Configure Model Optimizer
    printf "${dashes}"
    printf "Install Model Optimizer dependencies\n\n"
    cd "${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/install_prerequisites"
    . ./install_prerequisites.sh caffe
    cd "$cur_path"

    # Step 3. Convert a model with Model Optimizer
    printf "${dashes}"
    printf "Convert a model with Model Optimizer\n\n"

    mo_path="${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py"

    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
    print_and_run "$python_binary" "$downloader_dir/converter.py" --mo "$mo_path" --name "$model_name" -d "$models_path" -o "$irs_path" --precisions "$target_precision"
else
    printf "\n\nTarget folder ${ir_dir} already exists. Skipping IR generation  with Model Optimizer."
    printf "If you want to convert a model again, remove the entire ${ir_dir} folder. ${run_again}"
fi

# Step 4. Build samples
printf "${dashes}"
printf "Build Inference Engine samples\n\n"

OS_PATH=$(uname -m)
NUM_THREADS="-j2"

if [ "$OS_PATH" == "x86_64" ]; then
  OS_PATH="intel64"
  NUM_THREADS="-j8"
fi

samples_path="${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/samples/cpp"
build_dir="$HOME/inference_engine_cpp_samples_build"
binaries_dir="${build_dir}/${OS_PATH}/Release"

if [ -e "$build_dir/CMakeCache.txt" ]; then
  rm -rf "$build_dir/CMakeCache.txt"
fi
mkdir -p "$build_dir"
cd "$build_dir"
cmake -DCMAKE_BUILD_TYPE=Release "$samples_path"

make $NUM_THREADS classification_sample_async

# Step 5. Run samples
printf "${dashes}"
printf "Run Inference Engine classification sample\n\n"

cd "$binaries_dir"

cp -f "$ROOT_DIR/${model_name}.labels" "${ir_dir}/"

print_and_run ./classification_sample_async -d "$target" -i "$target_image_path" -m "${ir_dir}/${model_name}.xml" ${sampleoptions}

printf "${dashes}"

printf "Demo completed successfully.\n\n"
