// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"

#include <ngraph/function.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "generic_ie.hpp"

#include "legacy/convert_function_to_cnn_network.hpp"

using namespace ngraph;

TEST_F(NGraphReaderTests, ReadNonMaxSuppression5) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,15130,4"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>15130</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,80,15130"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>80</dim>
                    <dim>15130</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="max_output_boxes_per_class" precision="I64" type="Const" version="opset1">
            <data offset="0" size="8"/>
            <output>
                <port id="0" precision="I64"/>
            </output>
        </layer>
        <layer id="3" name="iou_threshold" precision="FP32" type="Const" version="opset1">
            <data offset="8" size="4"/>
            <output>
                <port id="0" precision="FP32"/>
            </output>
        </layer>
        <layer id="4" name="score_threshold" precision="FP32" type="Const" version="opset1">
            <data offset="12" size="4"/>
            <output>
                <port id="0" precision="FP32"/>
            </output>
        </layer>
        <layer id="5" name="soft_nms_sigma" precision="FP32" type="Const" version="opset1">
            <data offset="16" size="4"/>
            <output>
                <port id="0" precision="FP32"/>
            </output>
        </layer>
        <layer id="6" name="nms" type="NonMaxSuppression" version="opset5">
            <data box_encoding="corner" sort_result_descending="0"/>
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>15130</dim>
                    <dim>4</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>80</dim>
                    <dim>15130</dim>
                </port>
                <port id="2" precision="I64"/>
                <port id="3" precision="FP32"/>
                <port id="4" precision="FP32"/>
                <port id="5" precision="FP32"/>
            </input>
            <output>
                <port id="6" precision="I64">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
                <port id="7" precision="FP32">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
                <port id="8" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="7" name="mul" type="Multiply" version="opset1">
            <input>
                <port id="0" precision="I64">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="I64">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="8" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="I64">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
            </input>
        </layer>
        <layer id="9" name="mul2" type="Multiply" version="opset1">
            <input>
                <port id="0" precision="I64">
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="10" name="output2" type="Result" version="opset1">
            <input>
                <port id="0" precision="I64">
                    <dim>1</dim>
                </port>
            </input>
        </layer>
        <layer id="11" name="mul3" type="Multiply" version="opset1">
            <input>
                <port id="0" precision="I64">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="I64">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="12" name="output3" type="Result" version="opset1">
            <input>
                <port id="0" precision="I64">
                    <dim>15130</dim>
                    <dim>3</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="6" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="6" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="6" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="6" to-port="3"/>
        <edge from-layer="4" from-port="0" to-layer="6" to-port="4"/>
        <edge from-layer="5" from-port="0" to-layer="6" to-port="5"/>
        <edge from-layer="6" from-port="6" to-layer="7" to-port="0"/>
        <edge from-layer="6" from-port="6" to-layer="7" to-port="1"/>
        <edge from-layer="6" from-port="7" to-layer="11" to-port="0"/>
        <edge from-layer="6" from-port="7" to-layer="11" to-port="1"/>
        <edge from-layer="6" from-port="8" to-layer="9" to-port="0"/>
        <edge from-layer="6" from-port="8" to-layer="9" to-port="1"/>
        <edge from-layer="7" from-port="2" to-layer="8" to-port="0"/>
        <edge from-layer="9" from-port="2" to-layer="10" to-port="0"/>
        <edge from-layer="11" from-port="2" to-layer="12" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Network" version="7">
    <layers>
        <layer id="0" name="in1" precision="FP32" type="Input" >
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>15130</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" precision="FP32" type="Input" >
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>80</dim>
                    <dim>15130</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="max_output_boxes_per_class" precision="I64" type="Const">
            <output>
                <port id="0" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" precision="I64" size="8"/>
            </blobs>
        </layer>
        <layer id="3" name="iou_threshold" precision="FP32" type="Const">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="8" precision="FP32" size="4"/>
            </blobs>
        </layer>
        <layer id="4" name="score_threshold" precision="FP32" type="Const">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="12" precision="FP32" size="4"/>
            </blobs>
        </layer>
        <layer id="5" name="nms" type="NonMaxSuppression" precision="I64">
            <data center_point_box="false" output_type="I64" sort_result_descending="false"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>15130</dim>
                    <dim>4</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>80</dim>
                    <dim>15130</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="5" precision="I64">
                    <dim>16000</dim>
                    <dim>3</dim>
                </port>
                <port id="6" precision="FP32">
                    <dim>16000</dim>
                    <dim>3</dim>
                </port>
                <port id="7" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="mul" type="Eltwise" precision="I64">
            <data operation="prod"/>
            <input>
                <port id="0">
                    <dim>16000</dim>
                    <dim>3</dim>
                </port>
                <port id="1">
                    <dim>16000</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="I64">
                    <dim>16000</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="7" name="mul2" type="Eltwise" precision="I64">
            <data operation="prod"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="8" name="mul3" type="Eltwise" precision="FP32">
            <data operation="prod"/>
            <input>
                <port id="0">
                    <dim>16000</dim>
                    <dim>3</dim>
                </port>
                <port id="1">
                    <dim>16000</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>16000</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="5" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="5" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="5" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="5" to-port="3"/>
        <edge from-layer="4" from-port="0" to-layer="5" to-port="4"/>
        <edge from-layer="5" from-port="5" to-layer="6" to-port="0"/>
        <edge from-layer="5" from-port="5" to-layer="6" to-port="1"/>
        <edge from-layer="5" from-port="6" to-layer="8" to-port="0"/>
        <edge from-layer="5" from-port="6" to-layer="8" to-port="1"/>
        <edge from-layer="5" from-port="7" to-layer="7" to-port="0"/>
        <edge from-layer="5" from-port="7" to-layer="7" to-port="1"/>
    </edges>
</net>
)V0G0N";

    compareIRs(model, modelV5, 20, [](Blob::Ptr& weights) {
        auto * i64w = weights->buffer().as<int64_t*>();
        i64w[0] = 200;

        auto * fp32w = weights->buffer().as<float*>();
        fp32w[2] = 0.5;
        fp32w[3] = 0.05;
        fp32w[4] = 0.0;
    });
}

TEST_F(NGraphReaderTests, ReadNonMaxSuppression4) {
   std::string model = R"V0G0N(
<net name="Network" version="10">
   <layers>
       <layer id="0" name="in1" type="Parameter" version="opset1">
           <data element_type="f32" shape="1,15130,4"/>
           <output>
               <port id="0" precision="FP32">
                   <dim>1</dim>
                   <dim>15130</dim>
                   <dim>4</dim>
               </port>
           </output>
       </layer>
       <layer id="1" name="in2" type="Parameter" version="opset1">
           <data element_type="f32" shape="1,80,15130"/>
           <output>
               <port id="0" precision="FP32">
                   <dim>1</dim>
                   <dim>80</dim>
                   <dim>15130</dim>
               </port>
           </output>
       </layer>
       <layer id="2" name="max_output_boxes_per_class" precision="I64" type="Const" version="opset1">
           <data offset="0" size="8"/>
           <output>
               <port id="0" precision="I64"/>
           </output>
       </layer>
       <layer id="3" name="iou_threshold" precision="FP32" type="Const" version="opset1">
           <data offset="8" size="4"/>
           <output>
               <port id="0" precision="FP32"/>
           </output>
       </layer>
       <layer id="4" name="score_threshold" precision="FP32" type="Const" version="opset1">
           <data offset="12" size="4"/>
           <output>
               <port id="0" precision="FP32"/>
           </output>
       </layer>
       <layer id="5" name="nms" type="NonMaxSuppression" version="opset4">
           <data box_encoding="corner" output_type="i32" sort_result_descending="0"/>
           <input>
               <port id="0" precision="FP32">
                   <dim>1</dim>
                   <dim>15130</dim>
                   <dim>4</dim>
               </port>
               <port id="1" precision="FP32">
                   <dim>1</dim>
                   <dim>80</dim>
                   <dim>15130</dim>
               </port>
               <port id="2" precision="I64"/>
               <port id="3" precision="FP32"/>
               <port id="4" precision="FP32"/>
           </input>
           <output>
               <port id="5" precision="I32">
                   <dim>16000</dim>
                   <dim>3</dim>
               </port>
           </output>
       </layer>
       <layer id="6" name="mul" type="Multiply" version="opset1">
           <input>
               <port id="0" precision="I32">
                   <dim>16000</dim>
                   <dim>3</dim>
               </port>
               <port id="1" precision="I32">
                   <dim>16000</dim>
                   <dim>3</dim>
               </port>
           </input>
           <output>
               <port id="2" precision="I32">
                   <dim>16000</dim>
                   <dim>3</dim>
               </port>
           </output>
       </layer>
       <layer id="7" name="output" type="Result" version="opset1">
           <input>
               <port id="0" precision="I32">
                   <dim>16000</dim>
                   <dim>3</dim>
               </port>
           </input>
       </layer>
   </layers>
   <edges>
       <edge from-layer="0" from-port="0" to-layer="5" to-port="0"/>
       <edge from-layer="1" from-port="0" to-layer="5" to-port="1"/>
       <edge from-layer="2" from-port="0" to-layer="5" to-port="2"/>
       <edge from-layer="3" from-port="0" to-layer="5" to-port="3"/>
       <edge from-layer="4" from-port="0" to-layer="5" to-port="4"/>
       <edge from-layer="5" from-port="5" to-layer="6" to-port="0"/>
       <edge from-layer="5" from-port="5" to-layer="6" to-port="1"/>
       <edge from-layer="6" from-port="2" to-layer="7" to-port="0"/>
   </edges>
</net>
)V0G0N";

    constexpr size_t weightsSize = 16;

    Blob::Ptr weights;

    weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {weightsSize}, Layout::C));
    weights->allocate();
    CommonTestUtils::fill_data(weights->buffer().as<float *>(), weights->size() / sizeof(float));

    auto * i64w = weights->buffer().as<int64_t*>();
    i64w[0] = 200;

    auto * fp32w = weights->buffer().as<float*>();
    fp32w[2] = 0.5;
    fp32w[3] = 0.05;

    Core ie;

    auto ngraphImpl = ie.ReadNetwork(model, weights);
    auto graph = ngraph::clone_function(*ngraphImpl.getFunction());

    ::ngraph::pass::Manager manager;
    manager.register_pass<::ngraph::pass::InitNodeInfo>();
    manager.register_pass<::ngraph::pass::CommonOptimizations>();
    manager.run_passes(graph);

    auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 15130, 4});
    auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 80, 15130});
    auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {200});
    auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.5});
    auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.05});
    auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,  iou_threshold, score_threshold,
                                                           opset5::NonMaxSuppression::BoxEncodingType::CORNER, true, ngraph::element::i32);

    auto mul = std::make_shared<opset5::Multiply>(nms, nms);
    auto graph_ref = std::make_shared<Function>(NodeVector{mul}, ParameterVector{boxes, scores});

    auto res = compare_functions(graph, graph_ref);
    ASSERT_TRUE(res.first) << res.second;
}
