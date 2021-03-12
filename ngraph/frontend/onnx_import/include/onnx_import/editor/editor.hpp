//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <istream>
#include <map>
#include <memory>
#include <vector>

#include "ngraph/op/constant.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "onnx_import/core/operator_set.hpp"
#include "onnx_import/editor/detail/subgraph_extraction.hpp"
#include "onnx_import/utils/onnx_importer_visibility.hpp"

namespace ONNX_NAMESPACE
{
    // forward declaration to avoid the necessity of include paths setting in components
    // that don't directly depend on the ONNX library
    class ModelProto;
} // namespace ONNX_NAMESPACE

namespace ngraph
{
    namespace onnx_import
    {
        /// \brief A class representing a set of utilities allowing modification of an ONNX model
        ///
        /// \note This class can be used to modify an ONNX model before it gets translated to
        ///       an ngraph::Function by the import_onnx_model function. It lets you modify the
        ///       model's input types and shapes, extract a subgraph and more. An instance of this
        ///       class can be passed directly to the onnx_importer API.
        class ONNX_IMPORTER_API ONNXModelEditor final
        {
        public:
            ONNXModelEditor() = delete;

            /// \brief Creates an editor from a model file located on a storage device. The file
            ///        is parsed and loaded into the m_model_proto member variable.
            ///
            /// \param model_path Path to the file containing the model.
            ONNXModelEditor(const std::string& model_path);

            /// \brief Modifies the in-memory representation of the model by setting
            ///        custom input types for all inputs specified in the provided map.
            ///
            /// \param input_types A collection of pairs {input_name: new_input_type} that should be
            ///                    used to modified the ONNX model loaded from a file. This method
            ///                    throws an exception if the model doesn't contain any of
            ///                    the inputs specified in its parameter.
            void set_input_types(const std::map<std::string, element::Type_t>& input_types);

            /// \brief Modifies the in-memory representation of the model by setting
            ///        custom input shapes for all inputs specified in the provided map.
            ///
            /// \param input_shapes A collection of pairs {input_name: new_input_shape} that should
            ///                     be used to modified the ONNX model loaded from a file. This
            ///                     method throws an exception if the model doesn't contain any of
            ///                     the inputs specified in its parameter.
            void set_input_shapes(const std::map<std::string, ngraph::PartialShape>& input_shapes);

            /// \brief Extracts a subgraph constrained by input edges and output edges. In the end
            ///        the underlying ModelProto is modified - obsolete inputs, initializers, nodes
            ///        and outputs are removed from the in-memory model.
            ///
            /// \node Please look at the declaration of InputEdge and OutputEdge for explanation
            ///       how those objects can be created. If the outputs parameter is empty
            ///       this method keeps all of the original outputs of the model.
            ///
            /// \param inputs A collection of input edges which become new inputs to the graph
            /// \param outputs A collection of output edges which become new outputs of the graph
            void cut_graph_fragment(const std::vector<InputEdge>& inputs,
                                    const std::vector<OutputEdge>& outputs);
            /// \brief Modifies the in-memory representation of the model by setting custom input
            ///        values for inputs specified in the provided map.
            ///
            /// \note This method modifies existing initializer tensor if its name matches one of
            ///       input_name. Otherwise it adds initializer tensor into the model.
            ///       If input tensor of matching name is present in the model, its type and shape
            ///       are modified accordingly.
            ///
            /// \param input_values A collection of pairs {input_name: new_input_values} used to
            ///                     update the ONNX model. Initializers already existing are
            ///                     overwritten.
            void set_input_values(
                const std::map<std::string, std::shared_ptr<ngraph::op::Constant>>& input_values);

            /// \brief Returns a non-const reference to the underlying ModelProto object, possibly
            ///        modified by the editor's API calls
            ///
            /// \return A reference to ONNX ModelProto object containing the in-memory model
            ONNX_NAMESPACE::ModelProto& model() const;

            /// \brief Returns a serialized ONNX model, possibly modified by the editor.
            std::string model_string() const;

            /// \brief Returns a list of all inputs of the in-memory model, including initializers.
            ///        The returned value might depend on the previous operations executed on an
            ///        instance of the model editor, in particular the subgraph extraction which
            ///        can discard some inputs and initializers from the original graph.
            std::vector<std::string> model_inputs() const;

            /// \brief Returns the path to the original model file
            const std::string& model_path() const;

            /// \brief Saves the possibly modified model held by this class to a file.
            /// Serializes in binary mode.
            ///
            /// \param out_file_path A path to the file where the modified model should be dumped.
            void serialize(const std::string& out_file_path) const;

            /// \brief Replace nodes with given indexes with a newly registered custom operation
            ///
            /// \note custom op - a newly created onnx placeholder operation,
            ///                   after import will be replaced with the graph
            ///                   produced by node_generator function
            ///
            /// \note op_type for newly created custom ops is: custom_op_<ID>.
            ///       ID is an integer incremented with each call of this function (starts at 0).
            ///
            /// \param node_indexes A vector of index vectors which contains information
            ///                     which nodes should be replaced.
            ///                     - For each index vector, a new custom op will be inserted at
            ///                     the first index given.
            ///                     - Order of custom op inputs and outputs depends on
            ///                     node index order in index vector.
            ///                     - Provided index vectors must be a disjoint sets to each other,
            ///                     otherwise behaviour is undefinied.
            /// \param node_generator A function which returns a graph of nGraph nodes,
            ///                       which will replace the selected nodes in the original graph.
            void replace_nodes(std::vector<std::vector<int>> node_indexes, Operator node_generator);

        private:
            const std::string m_model_path;
            const std::string m_editor_domain{"org.openvinotoolkit.editor"};

            struct Impl;
            std::unique_ptr<Impl, void (*)(Impl*)> m_pimpl;

            int m_custom_op_ID = 0;

            void replace_nodes(std::vector<int>& node_indexes, const std::string& new_op_name);

            /// \brief Removes all nodes from a model whose index is in nodes_to_remove
            void remove_nodes(const std::vector<int>& nodes_to_remove);
        };
    } // namespace onnx_import
} // namespace ngraph
