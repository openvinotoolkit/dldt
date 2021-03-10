// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/op/op.hpp>
#include "store.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface ScalarStore
 * @brief Generated by Canonicalization for a scalar value store from vector register
 * @ingroup snippets
 */
class TRANSFORMATIONS_API ScalarStore : public Store {
public:
    NGRAPH_RTTI_DECLARATION;

    ScalarStore(const Output<Node>& x);
    ScalarStore() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<ScalarStore>(new_args.at(0));
    }
};

} // namespace op
} // namespace snippets
} // namespace ngraph