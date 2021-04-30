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

#include <frontend_manager/frontend_manager.hpp>
#include "exceptions.hpp"
#include "model.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph {
namespace frontend {

class NGRAPH_API FrontEndPDPD : public FrontEnd
{
    static std::shared_ptr<Function> convert_model(const std::shared_ptr<InputModelPDPD>& model);
    static std::shared_ptr<Node> make_const_node(const std::shared_ptr<TensorPlacePDPD>& place,
                                                 const std::shared_ptr<InputModelPDPD>& model);
public:

    FrontEndPDPD () { PDPD_CHECK(false, "FrontEnd failed"); };

    InputModel::Ptr loadFromFile (const std::string& path) const override
    {
        return std::make_shared<InputModelPDPD>(path);
    }

    std::shared_ptr<Function> convert (InputModel::Ptr model) const override;
};

} // namespace frontend
} // namespace ngraph
