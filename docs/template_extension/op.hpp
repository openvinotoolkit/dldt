// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>

#include "visibility.hpp"

//! [op:header]
namespace TemplateExtension {

class EXPORT Operation : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"Template", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }

    Operation() = default;
    Operation(const ngraph::Output<ngraph::Node>& arg, int64_t add);
    void validate_and_infer_types() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
    int64_t getAddAttr() { return add; }

private:
    int64_t add;
};
//! [op:header]

}  // namespace TemplateExtension
