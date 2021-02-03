#!/bin/bash

# Copyright (C) 2018-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

error() {
    local code="${3:-1}"
    if [[ -n "$2" ]];then
        echo "Error on or near line $1: $2; exiting with status ${code}"
    else
        echo "Error on or near line $1; exiting with status ${code}"
    fi
    exit "${code}"
} 
trap 'error ${LINENO}' ERR


V_ENV=0

for ((i=1;i <= $#;i++)) {
    case "${!i}" in
        caffe|tf|tf2|mxnet|kaldi|onnx)
            postfix="_$1"
            ;;
        "venv")
            V_ENV=1
            ;;
        *)
            if [[ "$1" != "" ]]; then
                echo "\"${!i}\" is unsupported parameter"
                echo $"Usage: $0 {caffe|tf|tf2|mxnet|kaldi|onnx} {venv}"
                exit 1
            fi
            ;;
        esac
}

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ -f /etc/centos-release ]]; then
    DISTRO="centos"
elif [[ -f /etc/lsb-release ]]; then
    DISTRO="ubuntu"
fi

if [[ $DISTRO == "centos" ]]; then
    if command -v python3.8 >/dev/null 2>&1; then
        python_binary=python3.8
    elif command -v python3.7 >/dev/null 2>&1; then
        python_binary=python3.7
    elif command -v python3.6 >/dev/null 2>&1; then
        python_binary=python3.6
    elif command -v python3.5 >/dev/null 2>&1; then
        python_binary=python3.5
    fi

    if [ -z "$python_binary" ]; then
        sudo -E yum install -y https://centos7.iuscommunity.org/ius-release.rpm
        sudo -E yum install -y python36u python36u-pip
        sudo -E pip3.6 install virtualenv
        python_binary=python3.6
    fi
    # latest pip is needed to install tensorflow
    sudo -E "$python_binary" -m pip install --upgrade pip
elif [[ $DISTRO == "ubuntu" ]]; then
    sudo -E apt update
    sudo -E apt -y --no-install-recommends install python3-pip python3-venv
    python_binary=python3
    sudo -E "$python_binary" -m pip install --upgrade pip
elif [[ "$OSTYPE" == "darwin"* ]]; then
    python_binary=python3
    python3 -m pip install --upgrade pip
fi

find_ie_bindings() {
    # check ie dependency
    echo "[install_prerequisites] Check IE"
    $1 "$SCRIPTDIR/../mo/utils/find_ie_version.py" $$ error=false || error=true
    if [ $error ]; then
        #install OpenVINO pip version
        echo "[install_prerequisites] Install IE"
        if [ $2 ]; then
            sudo -E $1 -m pip install openvino
        else
            $1 -m pip install openvino
        fi

        echo "[install_prerequisites] Check IE again"
        $1 "$SCRIPTDIR/../mo/utils/find_ie_version.py" $$ error=false || error=true
        if [ $error ]; then
            echo "[WARNING] No compatible OpenVINO python version was found"
        fi
    fi
}

if [[ $V_ENV -eq 1 ]]; then
    "$python_binary" -m venv "$SCRIPTDIR/../venv${postfix}"
    source "$SCRIPTDIR/../venv${postfix}/bin/activate"
    venv_python_binary="$SCRIPTDIR/../venv${postfix}/bin/$python_binary"
    $venv_python_binary -m pip install -r "$SCRIPTDIR/../requirements${postfix}.txt"
    find_ie_bindings $venv_python_binary
    echo
    echo "Before running the Model Optimizer, please activate virtualenv environment by running \"source ${SCRIPTDIR}/../venv${postfix}/bin/activate\""
else
    if [[ "$OSTYPE" == "darwin"* ]]; then
        python3 -m pip install -r "$SCRIPTDIR/../requirements${postfix}.txt"
        find_ie_bindings python3
    else
        sudo -E $python_binary -m pip install -r "$SCRIPTDIR/../requirements${postfix}.txt"
        find_ie_bindings $python_binary true
    fi
    echo "[WARNING] All Model Optimizer dependencies are installed globally."
    echo "[WARNING] If you want to keep Model Optimizer in separate sandbox"
    echo "[WARNING] run install_prerequisites.sh \"{caffe|tf|tf2|mxnet|kaldi|onnx}\" venv"
fi
