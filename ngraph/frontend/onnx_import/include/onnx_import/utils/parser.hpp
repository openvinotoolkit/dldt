//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once
#include <fstream>
#include <string>

namespace ONNX_NAMESPACE
{
    class ModelProto;
}

namespace ngraph
{
    namespace onnx_import
    {
        /// \brief   Parses an ONNX model from a file located on a storage device.
        ///
        /// \param   file_path    Path to the file containing an ONNX model.
        /// \param   model_proto  A reference to the object which will be filled with
        ///                           the model data
        void parse_from_file(const std::string& file_path, ONNX_NAMESPACE::ModelProto& model_proto);

        /// \brief   Parses an ONNX model from a stream (representing for example a file)
        ///
        /// \param   model_stream  Path to the file containing an ONNX model.
        /// \param   model_proto   A reference to the object which will be filled with
        ///                            the model data
        void parse_from_istream(std::istream& model_stream,
                                ONNX_NAMESPACE::ModelProto& model_proto);

        /// \brief   Opens an ifstream for a given model file.
        /// \param   file_path  Path to the file containing an ONNX model.
        ///
        /// \return  An object of std::ifstream for a given model.
        std::ifstream open_model_stream(const std::string& file_path);
    } // namespace onnx_import

} // namespace ngraph
