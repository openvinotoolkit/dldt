// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "low_precision/rt_info/precision_preserved_attribute.hpp"

class AvgPoolPrecisionPreservedAttribute {
public:
    AvgPoolPrecisionPreservedAttribute(std::shared_ptr<PrecisionPreservedAttribute::SharedValue> sharedValue) : sharedValue(sharedValue) {}

    template <class Operation>
    static std::shared_ptr<AvgPoolPrecisionPreservedAttribute> create(const bool value) {
        // TODO: do we need operation version here?
        auto operationName = Operation::get_type_info_static().name;
        return std::make_shared<AvgPoolPrecisionPreservedAttribute>(value, operationName);
    }

    std::shared_ptr<PrecisionPreservedAttribute::SharedValue> sharedValue;
};

using AvgPoolPrecisionPreservedAttributePtr = std::shared_ptr<AvgPoolPrecisionPreservedAttribute>;

extern template class TRANSFORMATIONS_API ngraph::VariantImpl<AvgPoolPrecisionPreservedAttributePtr>;

template<>
class TRANSFORMATIONS_API ngraph::VariantWrapper<AvgPoolPrecisionPreservedAttributePtr> : public ngraph::VariantImpl<AvgPoolPrecisionPreservedAttributePtr> {
public:
    static constexpr ngraph::VariantTypeInfo type_info{ "AVG_POOL_PRECISION_PRESERVED", 0 };

    const ngraph::VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector& nodes) override;

    AvgPoolPrecisionPreservedAttributePtr get() { return this->m_value; }

    std::string get_string() override;
};
