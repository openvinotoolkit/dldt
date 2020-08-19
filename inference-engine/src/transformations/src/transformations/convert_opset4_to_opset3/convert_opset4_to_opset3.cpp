// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset4_to_opset3/convert_opset4_to_opset3.hpp"
#include "transformations/convert_opset4_to_opset3/convert_sequences_to_sequences_ie.hpp"
#include "transformations/convert_opset4_to_opset3/convert_tensor_iterator_to_sequence.hpp"
#include "transformations/itt.hpp"

#include <memory>
#include <vector>

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

bool ngraph::pass::ConvertOpSet4ToOpSet3::run_on_function(std::shared_ptr<ngraph::Function> f) {
    OV_ITT_SCOPED_TASK(itt::domains::IETransform, "ngraph::pass::ConvertOpSet4ToOpSet3");

    ngraph::pass::Manager manager;

    manager.register_pass<ngraph::pass::ConvertTensorIteratorToLSTMSequence>();
    manager.register_pass<ngraph::pass::ConvertTensorIteratorToGRUSequence>();
    manager.register_pass<ngraph::pass::ConvertTensorIteratorToRNNSequence>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertGRUSequenceMatcher>();
    manager.register_pass<ngraph::pass::ConvertRNNSequenceMatcher>();
    manager.register_pass<ngraph::pass::ConvertLSTMSequenceMatcher>();
    manager.register_pass<ngraph::pass::ConstantFolding>();

    manager.set_callback(m_transformation_callback);
    manager.run_passes(f);
    return true;
}
