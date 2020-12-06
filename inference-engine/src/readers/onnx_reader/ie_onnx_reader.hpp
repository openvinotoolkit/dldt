// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_reader.hpp>

namespace InferenceEngineONNXReader {

using namespace InferenceEngine;

INFERENCE_PLUGIN_API(StatusCode) CreateReader(IReader*& reader, ResponseDesc *resp) noexcept;

class ONNXReader: public IReader {
public:
    void Release() noexcept override {
        delete this;
    }
    /**
     * @brief Checks that reader supports format of the model
     * @param model stream with model
     * @return true if format is supported
     */
    bool supportModel(std::istream& model) const override;
    /**
     * @brief Reads the model to CNNNetwork
     * @param model stream with model
     * @param exts vector with extensions
     *
     * @return CNNNetwork
     */
    CNNNetwork read(std::istream& model, const std::vector<IExtensionPtr>& exts) const override;
    /**
     * @brief Reads the model to CNNNetwork
     * @param model stream with model
     * @param weights blob with binary data
     * @param exts vector with extensions
     *
     * @return CNNNetwork
     */
    CNNNetwork read(std::istream& model, const Blob::CPtr& weights, const std::vector<IExtensionPtr>& exts) const override {
        THROW_IE_EXCEPTION << "ONNX reader cannot read model with weights!";
    }

    std::vector<std::string> getDataFileExtensions() const override {
        return {};
    }
};

}  // namespace InferenceEngineONNXReader
