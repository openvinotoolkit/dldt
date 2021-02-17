# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
#
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

import os
import pathlib
import shutil
import glob
import sysconfig
import sys
import multiprocessing

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop
from distutils.command.build import build as _build

__version__ = os.environ.get("NGRAPH_VERSION", "0.0.0.dev0")
PYNGRAPH_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
NGRAPH_ROOT_DIR = os.path.normpath(os.path.join(PYNGRAPH_ROOT_DIR, ".."))
OPENVINO_ROOT_DIR = os.path.normpath(os.path.join(PYNGRAPH_ROOT_DIR, "../.."))
# Change current working dircectory to ngraph/python
os.chdir(PYNGRAPH_ROOT_DIR)


# debug_optimization_flags = [
#     "O1", "O2", "O3", "O4", "Ofast", "Os", "Oz", "Og", "O", "DNDEBUG"
# ]
#
#
# def find_ngraph_dist_dir():
#     """Return location of compiled ngraph library home."""
#     if os.environ.get("NGRAPH_CPP_BUILD_PATH"):
#         ngraph_dist_dir = os.environ.get("NGRAPH_CPP_BUILD_PATH")
#     else:
#         ngraph_dist_dir = os.path.join(NGRAPH_DEFAULT_INSTALL_DIR, "ngraph_dist")
#
#     found = os.path.exists(os.path.join(ngraph_dist_dir, "include/ngraph"))
#     if not found:
#         print(
#             "Cannot find nGraph library in {} make sure that "
#             "NGRAPH_CPP_BUILD_PATH is set correctly".format(ngraph_dist_dir)
#         )
#         sys.exit(1)
#     else:
#         print("nGraph library found in {}".format(ngraph_dist_dir))
#         return ngraph_dist_dir
#
#
# def find_pybind_headers_dir():
#     """Return location of pybind11 headers."""
#     if os.environ.get("PYBIND_HEADERS_PATH"):
#         pybind_headers_dir = os.environ.get("PYBIND_HEADERS_PATH")
#     else:
#         pybind_headers_dir = os.path.join(PYNGRAPH_ROOT_DIR, "pybind11")
#
#     found = os.path.exists(os.path.join(pybind_headers_dir, "include/pybind11"))
#     if not found:
#         print(
#             "Cannot find pybind11 library in {} make sure that "
#             "PYBIND_HEADERS_PATH is set correctly".format(pybind_headers_dir)
#         )
#         sys.exit(1)
#     else:
#         print("pybind11 library found in {}".format(pybind_headers_dir))
#         return pybind_headers_dir
#
#
# OPENVINO_DIST_PATH = os.environ.get("OPENVINO_DIST_PATH")
# IE_CPP_INCLUDE_DIR = os.path.join(OPENVINO_DIST_PATH, "deployment_tools", "inference_engine", "include")
# PYBIND11_INCLUDE_DIR = find_pybind_headers_dir() + "/include"
# NGRAPH_CPP_DIST_DIR = os.path.join(OPENVINO_DIST_PATH, "deployment_tools", "ngraph")
# NGRAPH_CPP_INCLUDE_DIR = os.path.join(NGRAPH_CPP_DIST_DIR, "include")
# if os.path.exists(os.path.join(NGRAPH_CPP_DIST_DIR, "lib")):
#     NGRAPH_CPP_LIBRARY_DIR = os.path.join(NGRAPH_CPP_DIST_DIR, "lib")
# elif os.path.exists(os.path.join(NGRAPH_CPP_DIST_DIR, "lib64")):
#     NGRAPH_CPP_LIBRARY_DIR = os.path.join(NGRAPH_CPP_DIST_DIR, "lib64")
# else:
#     print(
#         "Cannot find library directory in {}, make sure that nGraph is installed "
#         "correctly".format(NGRAPH_CPP_DIST_DIR)
#     )
#     sys.exit(1)
#
# if sys.platform == "win32":
#     NGRAPH_CPP_DIST_DIR = os.path.normpath(NGRAPH_CPP_DIST_DIR)
#     PYBIND11_INCLUDE_DIR = os.path.normpath(PYBIND11_INCLUDE_DIR)
#     NGRAPH_CPP_INCLUDE_DIR = os.path.normpath(NGRAPH_CPP_INCLUDE_DIR)
#     NGRAPH_CPP_LIBRARY_DIR = os.path.normpath(NGRAPH_CPP_LIBRARY_DIR)
#
# NGRAPH_CPP_LIBRARY_NAME = "ngraph"
# """For some platforms OpenVINO adds 'd' suffix to library names in debug configuration"""
# if len([fn for fn in os.listdir(NGRAPH_CPP_LIBRARY_DIR) if re.search("ngraphd", fn)]):
#     NGRAPH_CPP_LIBRARY_NAME = "ngraphd"
#
# ONNX_IMPORTER_CPP_LIBRARY_NAME = "onnx_importer"
# if len([fn for fn in os.listdir(NGRAPH_CPP_LIBRARY_DIR) if re.search("onnx_importerd", fn)]):
#     ONNX_IMPORTER_CPP_LIBRARY_NAME = "onnx_importerd"
#
#
# def _remove_compiler_flags(obj):
#     """Make pybind11 more verbose in debug builds."""
#     for flag in debug_optimization_flags:
#         try:
#             if sys.platform == "win32":
#                 obj.compiler.compile_options.remove("/{}".format(flag))
#             else:
#                 obj.compiler.compiler_so.remove("-{}".format(flag))
#                 obj.compiler.compiler.remove("-{}".format(flag))
#         except (AttributeError, ValueError):
#             pass
#
#
# def parallelCCompile(
#     self,
#     sources,
#     output_dir=None,
#     macros=None,
#     include_dirs=None,
#     debug=0,
#     extra_preargs=None,
#     extra_postargs=None,
#     depends=None,
# ):
#     """Build sources in parallel.
#
#     Reference link:
#     http://stackoverflow.com/questions/11013851/speeding-up-build-process-with-distutils
#     Monkey-patch for parallel compilation.
#     """
#     # those lines are copied from distutils.ccompiler.CCompiler directly
#     macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
#         output_dir, macros, include_dirs, sources, depends, extra_postargs
#     )
#     cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
#
#     # parallel code
#     import multiprocessing.pool
#
#     def _single_compile(obj):
#         try:
#             src, ext = build[obj]
#         except KeyError:
#             return
#         self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
#
#     # convert to list, imap is evaluated on-demand
#     pool = multiprocessing.pool.ThreadPool()
#     list(pool.imap(_single_compile, objects))
#     return objects
#
#
# ccompiler.compile = parallelCCompile
#
#
# def has_flag(compiler, flagname):
#     """Check whether a flag is supported by the specified compiler.
#
#     As of Python 3.6, CCompiler has a `has_flag` method.
#     cf http://bugs.python.org/issue26689
#     """
#     import tempfile
#
#     with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
#         f.write("int main (int argc, char **argv) { return 0; }")
#         try:
#             compiler.compile([f.name], extra_postargs=[flagname])
#         except setuptools.distutils.errors.CompileError:
#             return False
#     return True
#
#
# def cpp_flag(compiler):
#     """Check and return the -std=c++11 compiler flag."""
#     if sys.platform == "win32":
#         return ""  # C++11 is on by default in MSVC
#     elif has_flag(compiler, "-std=c++11"):
#         return "-std=c++11"
#     else:
#         raise RuntimeError("Unsupported compiler -- C++11 support is needed!")
#
#
# sources = [
#     "pyngraph/axis_set.cpp",
#     "pyngraph/axis_vector.cpp",
#     "pyngraph/coordinate.cpp",
#     "pyngraph/coordinate_diff.cpp",
#     "pyngraph/dict_attribute_visitor.cpp",
#     "pyngraph/dimension.cpp",
#     "pyngraph/function.cpp",
#     "pyngraph/node.cpp",
#     "pyngraph/node_input.cpp",
#     "pyngraph/node_output.cpp",
#     "pyngraph/node_factory.cpp",
#     "pyngraph/ops/constant.cpp",
#     "pyngraph/ops/parameter.cpp",
#     "pyngraph/ops/result.cpp",
#     "pyngraph/ops/util/arithmetic_reduction.cpp",
#     "pyngraph/ops/util/binary_elementwise_arithmetic.cpp",
#     "pyngraph/ops/util/binary_elementwise_comparison.cpp",
#     "pyngraph/ops/util/binary_elementwise_logical.cpp",
#     "pyngraph/ops/util/index_reduction.cpp",
#     "pyngraph/ops/util/op_annotations.cpp",
#     "pyngraph/ops/util/regmodule_pyngraph_op_util.cpp",
#     "pyngraph/ops/util/unary_elementwise_arithmetic.cpp",
#     "pyngraph/passes/manager.cpp",
#     "pyngraph/passes/regmodule_pyngraph_passes.cpp",
#     "pyngraph/partial_shape.cpp",
#     "pyngraph/pyngraph.cpp",
#     "pyngraph/shape.cpp",
#     "pyngraph/strides.cpp",
#     "pyngraph/tensor_iterator_builder.cpp",
#     "pyngraph/types/element_type.cpp",
#     "pyngraph/types/regmodule_pyngraph_types.cpp",
#     "pyngraph/util.cpp",
#     "pyngraph/variant.cpp",
#     "pyngraph/rt_map.cpp",
# ]
# sources = [PYNGRAPH_SRC_DIR + "/" + source for source in sources]
#
# pyopenvino_sources = [
#     "pyopenvino/pyopenvino.cpp",
#     "pyopenvino/inference_engine/ie_core.cpp",
#     "pyopenvino/inference_engine/ie_executable_network.cpp",
#     "pyopenvino/inference_engine/ie_infer_request.cpp",
#     "pyopenvino/inference_engine/ie_network.cpp",
#     "pyopenvino/inference_engine/tensor_description.cpp",
#     "pyopenvino/inference_engine/ie_version.cpp",
#     "pyopenvino/inference_engine/ie_parameter.cpp",
#     "pyopenvino/inference_engine/ie_data.cpp",
#     "pyopenvino/inference_engine/ie_input_info.cpp",
#     "pyopenvino/inference_engine/ie_blob.cpp",
#     "pyopenvino/inference_engine/common.cpp",
# ]
# pyopenvino_sources = [PYNGRAPH_SRC_DIR + "/" + source for source in pyopenvino_sources]

NGRAPH_LIBS = ["ngraph", "onnx_importer", "inference_engine"]


packages = [
    "ngraph",
    "ngraph.opset1",
    "ngraph.opset2",
    "ngraph.opset3",
    "ngraph.opset4",
    "ngraph.opset5",
    "ngraph.opset6",
    "ngraph.utils",
    "ngraph.impl",
    "ngraph.impl.op",
    "ngraph.impl.op.util",
    "ngraph.impl.passes",
    "openvino",
    "openvino.inference_engine"
]

# include_dirs = [PYNGRAPH_SRC_DIR, NGRAPH_CPP_INCLUDE_DIR, IE_CPP_INCLUDE_DIR, PYBIND11_INCLUDE_DIR]
#
# IE_CPP_LIBRARY_DIR = os.path.join(OPENVINO_DIST_PATH, "deployment_tools", "inference_engine", "lib", "intel64")
#
# library_dirs = [NGRAPH_CPP_LIBRARY_DIR, IE_CPP_LIBRARY_DIR]

data_files = []

with open(os.path.join(PYNGRAPH_ROOT_DIR, "requirements.txt")) as req:
    requirements = req.read().splitlines()

cmdclass = {}
for super_class in [_build, _install, _develop]:
    class command(super_class):
        """Add user options for build, install and develop commands."""

        cmake_build_types = ["Release", "Debug", "RelWithDebInfo", "MinSizeRel"]
        user_options = super_class.user_options + [
            ("config=", None, "Build configuration [{}].".format("|".join(cmake_build_types))),
            ("jobs=", None, "Specifies the number of jobs to use with make."),
            ("cmake-args=", None, "Additional options to be passed to CMake.")
        ]
        def initialize_options(self):
            """Set default values for all the options that this command supports."""
            super().initialize_options()
            self.config = None
            self.jobs = None
            self.cmake_args = None

    cmdclass[super_class.__name__] = command

# print("NGRAPH_CPP_LIBRARY_NAME, ONNX_IMPORTER_CPP_LIBRARY_NAME", NGRAPH_CPP_LIBRARY_NAME, ONNX_IMPORTER_CPP_LIBRARY_NAME)
#
# ext_modules = [
#     Extension(
#         "_pyngraph",
#         sources=sources,
#         include_dirs=include_dirs,
#         define_macros=[("VERSION_INFO", __version__)],
#         library_dirs=library_dirs,
#         libraries=[NGRAPH_CPP_LIBRARY_NAME, ONNX_IMPORTER_CPP_LIBRARY_NAME],
#     ),
#     Extension(
#         "openvino.pyopenvino",
#         sources=pyopenvino_sources,
#         include_dirs=include_dirs,
#         library_dirs=library_dirs,
#         libraries=["inference_engine"],
#     ),
# ]

class CMakeExtension(Extension):
    """Build extension stub."""

    def __init__(self, name, sources=None):
        if sources is None:
            sources = []
        super().__init__(name=name, sources=sources)


class BuildCMakeExt(build_ext):
    """Builds module using cmake instead of the python setuptools implicit build."""

    cmake_build_types = ["Release", "Debug", "RelWithDebInfo", "MinSizeRel"]
    user_options = [
        ("config=", None, "Build configuration [{}].".format("|".join(cmake_build_types))),
        ("jobs=", None, "Specifies the number of jobs to use with make."),
        ("cmake-args=", None, "Additional options to be passed to CMake.")
    ]

    def initialize_options(self):
        """Set default values for all the options that this command supports."""
        super().initialize_options()
        self.build_base = "build"
        self.config = None
        self.jobs = None
        self.cmake_args = None

    def finalize_options(self):
        """Set final values for all the options that this command supports."""
        super().finalize_options()

        for cmd in ["build", "install", "develop"]:
            self.set_undefined_options(cmd, ("config", "config"),
                                       ("jobs", "jobs"),
                                       ("cmake_args", "cmake_args"))

        if not self.config:
            if self.debug:
                self.config = "Debug"
            else:
                self.announce("Set default value for CMAKE_BUILD_TYPE = Release.", level=4)
                self.config = "Release"
        else:
            build_types = [item.lower() for item in self.cmake_build_types]
            try:
                i = build_types.index(str(self.config).lower())
                self.config = self.cmake_build_types[i]
                self.debug = True if "Debug" == self.config else False
            except ValueError:
                self.announce("Unsupported CMAKE_BUILD_TYPE value: " + self.config, level=4)
                self.announce("Supported values: {}".format(", ".join(self.cmake_build_types)), level=4)
                sys.exit(1)
        if self.jobs is None and os.getenv("MAX_JOBS") is not None:
            self.jobs = os.getenv("MAX_JOBS")
        self.jobs = multiprocessing.cpu_count() if self.jobs is None else int(self.jobs)

    def run(self):
        """Run CMake build for modules."""
        for extension in self.extensions:
            if extension.name == "_pyngraph":
                self.build_cmake(extension)

    def build_cmake(self, extension: Extension):
        """Cmake configure and build steps."""
        self.announce("Preparing the build environment", level=3)
        plat_specifier = ".%s-%d.%d" % (self.plat_name, *sys.version_info[:2])
        self.build_temp = os.path.join(self.build_base, "temp" + plat_specifier, self.config)
        build_dir = pathlib.Path(self.build_temp)

        extension_path = pathlib.Path(self.get_ext_fullpath(extension.name))

        os.makedirs(build_dir, exist_ok=True)
        os.makedirs(extension_path.parent.absolute(), exist_ok=True)

        # If ngraph_DIR is not set try to build from OpenVINO root
        root_dir = OPENVINO_ROOT_DIR
        bin_dir = os.path.join(OPENVINO_ROOT_DIR, "bin")
        if os.environ.get("ngraph_DIR") is not None:
            root_dir = PYNGRAPH_ROOT_DIR
            bin_dir = build_dir

        self.announce("Configuring cmake project", level=3)
        ext_args = self.cmake_args.split() if self.cmake_args else []
        self.spawn(["cmake", "-H" + root_dir, "-B" + self.build_temp,
                    "-DCMAKE_BUILD_TYPE={}".format(self.config),
                    "-DNGRAPH_PYTHON_BUILD_ENABLE=ON",
                    "-DNGRAPH_ONNX_IMPORT_ENABLE=ON"] + ext_args)

        self.announce("Building binaries", level=3)

        self.spawn(["cmake", "--build", self.build_temp, "--target", extension.name,
                    "--config", self.config, "-j", str(self.jobs)])

        self.announce("Moving built python module to " + str(extension_path), level=3)
        pyds = list(glob.iglob("{0}/**/{1}*{2}".format(bin_dir,
                    extension.name,
                    sysconfig.get_config_var("EXT_SUFFIX")), recursive=True))
        for name in pyds:
            self.announce("copy " + os.path.join(name), level=3)
            shutil.copy(name, extension_path)


class InstallCMakeLibs(install_lib):
    """Finds and installs NGraph libraries to a package location."""

    def run(self):
        """Copy libraries from the bin directory and place them as appropriate."""
        self.announce("Adding library files", level=3)

        root_dir = os.path.join(OPENVINO_ROOT_DIR, "bin")
        if os.environ.get("ngraph_DIR") is not None:
            root_dir = pathlib.Path(os.environ["ngraph_DIR"]) / ".."

        lib_ext = ""
        if "linux" in sys.platform:
            lib_ext = ".so"
        elif sys.platform == "darwin":
            lib_ext = ".dylib"
        elif sys.platform == "win32":
            lib_ext = ".dll"

        libs = []
        for ngraph_lib in NGRAPH_LIBS:
            libs.extend(list(glob.iglob("{0}/**/*{1}*{2}".format(root_dir,
                             ngraph_lib, lib_ext), recursive=True)))
        if not libs:
            raise Exception("NGraph libs not found.")

        self.announce("Adding library files" + str(libs), level=3)

        self.distribution.data_files.extend([("lib", [os.path.normpath(lib) for lib in libs])])
        self.distribution.run_command("install_data")
        super().run()


cmdclass["build_ext"] = BuildCMakeExt
cmdclass["install_lib"] = InstallCMakeLibs

setup(
    name="ngraph-core",
    description="nGraph - Intel's graph compiler and runtime for Neural Networks",
    version=__version__,
    author="Intel Corporation",
    url="https://github.com/openvinotoolkit/openvino",
    license="License :: OSI Approved :: Apache Software License",
    ext_modules=[CMakeExtension(name="_pyngraph"), CMakeExtension(name="openvino.pyopenvino")],
    package_dir={"": "src"},
    packages=packages,
    install_requires=requirements,
    data_files=data_files,
    zip_safe=False,
    extras_require={},
    cmdclass=cmdclass
)
