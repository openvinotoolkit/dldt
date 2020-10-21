// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// ATTENTION :
// - dynamic shapes are not supported
// - GenericIE operation type (experimental opset) is not supported
// - order of generated layers in xml file is ngraph specific (given by
// get_ordered_ops()); MO generates file with different order, but they are
// logically equivalent

#pragma once

#include <string>

#include "ngraph/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API Serialize;

}  // namespace pass
}  // namespace ngraph

// ! [function_pass:serialize_hpp]
// serialize.hpp
class ngraph::pass::Serialize : public ngraph::pass::FunctionPass {
public:
    enum class Version { IR_V10 };
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

    Serialize(const std::string& xmlPath, const std::string& binPath,
              Version version = Version::IR_V10)
        : m_xmlPath{xmlPath}, m_binPath{binPath}, m_version{version} {}

private:
    const std::string m_xmlPath;
    const std::string m_binPath;
    const Version m_version;
};
// ! [function_pass:serialize_hpp]
