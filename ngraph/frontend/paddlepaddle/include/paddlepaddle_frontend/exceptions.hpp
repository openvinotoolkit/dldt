// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_manager.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {

class NodeContext;

class CheckFailurePDPD : public CheckFailureFrontEnd {
public:
    CheckFailurePDPD(const CheckLocInfo &check_loc_info, const std::string &context, const std::string &explanation)
            : CheckFailureFrontEnd(check_loc_info, " \nPaddlePaddle FrontEnd failed" + context, explanation) {
    }
};

class NodeValidationFailurePDPD : public CheckFailurePDPD {
public:
    NodeValidationFailurePDPD(const CheckLocInfo &check_loc_info,
                              const pdpd::NodeContext &node,
                              const std::string &explanation)
            : CheckFailurePDPD(check_loc_info, get_error_msg_prefix_pdpd(node), explanation) {
    }

private:
    static std::string get_error_msg_prefix_pdpd(const pdpd::NodeContext &node);
};
} // namespace pdpd
} // namespace frontend

/// \brief Macro to check whether a boolean condition holds.
/// \param node_context Object of NodeContext class
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ngraph::CheckFailurePDPD if `cond` is false.
#define PDPD_NODE_VALIDATION_CHECK(node_context, ...) \
        NGRAPH_CHECK_HELPER(::ngraph::frontend::pdpd::NodeValidationFailurePDPD, (node_context), __VA_ARGS__)

/// \brief Macro to check whether a boolean condition holds.
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ngraph::CheckFailurePDPD if `cond` is false.
#define PDPD_CHECK(...) NGRAPH_CHECK_HELPER(::ngraph::frontend::pdpd::CheckFailurePDPD, "", __VA_ARGS__)

#define PDPD_NOT_IMPLEMENTED(msg) PDPD_CHECK(false, std::string(msg) + " is not implemented")

} // namespace ngraph

