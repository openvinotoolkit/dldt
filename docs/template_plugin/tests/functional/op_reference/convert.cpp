// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <tuple>

#include "base_reference_test.hpp"

using namespace ngraph;
using namespace InferenceEngine;

struct ConvertParams {
    template <class IT, class OT>
    ConvertParams(const ngraph::PartialShape& shape, const ngraph::element::Type& iType, const ngraph::element::Type& oType, const std::vector<IT>& iValues,
                  const std::vector<OT>& oValues, size_t iSize = 0, size_t oSize = 0)
        : pshape(shape), inType(iType), outType(oType), inputData(CreateBlob(iType, iValues, iSize)), refData(CreateBlob(oType, oValues, oSize)) {}
    ngraph::PartialShape pshape;
    ngraph::element::Type inType;
    ngraph::element::Type outType;
    InferenceEngine::Blob::Ptr inputData;
    InferenceEngine::Blob::Ptr refData;
};

class ReferenceConvertLayerTest : public testing::TestWithParam<ConvertParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }

private:
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape, const element::Type& input_type,
                                                    const element::Type& expected_output_type) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape);
        const auto convert = std::make_shared<op::Convert>(in, expected_output_type);
        return std::make_shared<Function>(NodeVector {convert}, ParameterVector {in});
    }
};

TEST_P(ReferenceConvertLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Convert_With_Hardcided_Refs, ReferenceConvertLayerTest,
    ::testing::Values(
        // destination boolean
        ConvertParams(ngraph::PartialShape {2, 3}, ngraph::element::u8, ngraph::element::boolean,
                      std::vector<uint8_t> {0, 12, 23, 0, std::numeric_limits<uint8_t>::lowest(), std::numeric_limits<uint8_t>::max()},
                      std::vector<char> {0, 1, 1, 0, 0, 1}),
        ConvertParams(ngraph::PartialShape {2, 3}, ngraph::element::i32, ngraph::element::boolean,
                      std::vector<int32_t> {0, -12, 23, 0, std::numeric_limits<int32_t>::lowest(), std::numeric_limits<int32_t>::max()},
                      std::vector<char> {0, 1, 1, 0, 1, 1}),
        ConvertParams(ngraph::PartialShape {3, 3}, ngraph::element::f32, ngraph::element::boolean,
                      std::vector<float> {0.f, 1.5745f, 0.12352f, 0.f, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(),
                                          std::numeric_limits<float>::min(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()},
                      std::vector<char> {0, 1, 1, 0, 1, 1, 1, 1, 1}),

        // destination bf16
        ConvertParams(ngraph::PartialShape {1, 1, 3, 5}, ngraph::element::f32, ngraph::element::bf16,
                      std::vector<float> {0.5f, 1.5f, 0.5f, 2.5f, 1.5f, 0.5f, 3.5f, 2.5f, 0.5f, 0.5f, 2.5f, 0.5f, 0.5f, 0.5f, 1.5f},
                      std::vector<bfloat16> {0.5f, 1.5f, 0.5f, 2.5f, 1.5f, 0.5f, 3.5f, 2.5f, 0.5f, 0.5f, 2.5f, 0.5f, 0.5f, 0.5f, 1.5f}),
        ConvertParams(ngraph::PartialShape {11}, ngraph::element::u8, ngraph::element::bf16,
                      std::vector<uint8_t> {0, 10, 15, 20, 43, 56, 78, 99, 102, 130, 142},
                      std::vector<bfloat16> {0, 10, 15, 20, 43, 56, 78, 99, 102, 130, 142}),

        // destination f16
        ConvertParams(ngraph::PartialShape {1, 1, 3, 5}, ngraph::element::f32, ngraph::element::f16,
                      std::vector<float> {0.5f, 1.5f, 0.5f, 2.5f, 1.5f, 0.5f, 3.5f, 2.5f, 0.5f, 0.5f, 2.5f, 0.5f, 0.5f, 0.5f, 1.5f},
                      std::vector<float16> {0.5f, 1.5f, 0.5f, 2.5f, 1.5f, 0.5f, 3.5f, 2.5f, 0.5f, 0.5f, 2.5f, 0.5f, 0.5f, 0.5f, 1.5f}),
        ConvertParams(ngraph::PartialShape {11}, ngraph::element::u8, ngraph::element::f16, std::vector<uint8_t> {0, 10, 15, 20, 43, 56, 78, 99, 102, 130, 142},
                      std::vector<float16> {0, 10, 15, 20, 43, 56, 78, 99, 102, 130, 142}),

        // destination f32
        ConvertParams(ngraph::PartialShape {2, 2}, ngraph::element::u1, ngraph::element::f32, std::vector<uint8_t> {0xA0},
                      std::vector<float> {1.0f, 0.0f, 1.0f, 0.0f}, 4),
        ConvertParams(ngraph::PartialShape {2, 2}, ngraph::element::u4, ngraph::element::f32, std::vector<uint8_t> {0xFB, 0x0A},
                      std::vector<float> {15.0f, 11.0f, 0.0f, 10.0f}, 4),
        ConvertParams(ngraph::PartialShape {2, 2}, ngraph::element::u8, ngraph::element::f32, std::vector<uint8_t> {255, 128, 32, 0},
                      std::vector<float> {255.0f, 128.0f, 32.0f, 0.0f}),
        ConvertParams(ngraph::PartialShape {2, 2}, ngraph::element::u16, ngraph::element::f32, std::vector<uint16_t> {64000, 32000, 128, 0},
                      std::vector<float> {64000.0f, 32000.0f, 128.0f, 0.0f}),
        ConvertParams(ngraph::PartialShape {2, 2}, ngraph::element::u32, ngraph::element::f32, std::vector<uint32_t> {4000000, 2000000, 128, 0},
                      std::vector<float> {4000000.0f, 2000000.0f, 128.0f, 0.0f}),
        ConvertParams(ngraph::PartialShape {2, 2}, ngraph::element::u64, ngraph::element::f32, std::vector<uint64_t> {4000000, 2000000, 128, 0},
                      std::vector<float> {4000000.0f, 2000000.0f, 128.0f, 0.0f}),
        ConvertParams(ngraph::PartialShape {2, 2}, ngraph::element::i4, ngraph::element::f32, std::vector<uint8_t> {0xFE, 0xF2},
                      std::vector<float> {-1.0f, -2.0f, -1.0f, 2.0f}, 4),
        ConvertParams(ngraph::PartialShape {2, 2}, ngraph::element::i8, ngraph::element::f32, std::vector<int8_t> {-127, -0, 0, 127},
                      std::vector<float> {-127.0f, -0.0f, 0.0f, 127.0f}),
        ConvertParams(ngraph::PartialShape {2, 2}, ngraph::element::i16, ngraph::element::f32, std::vector<int16_t> {-32000, -0, 0, 32000},
                      std::vector<float> {-32000.0f, -0.0f, 0.0f, 32000.0f}),
        ConvertParams(ngraph::PartialShape {2, 2}, ngraph::element::i32, ngraph::element::f32, std::vector<int32_t> {-64000, -0, 0, 64000},
                      std::vector<float> {-64000.0f, -0.0f, 0.0f, 64000.0f}),
        ConvertParams(ngraph::PartialShape {2, 2}, ngraph::element::i64, ngraph::element::f32, std::vector<int64_t> {-64000, -0, 0, 64000},
                      std::vector<float> {-64000.0f, -0.0f, 0.0f, 64000.0f}),
        ConvertParams(ngraph::PartialShape {1, 1, 3, 5}, ngraph::element::bf16, ngraph::element::f32,
                      std::vector<bfloat16> {0.5f, 1.5f, 0.5f, 2.5f, 1.5f, 0.5f, 3.5f, 2.5f, 0.5f, 0.5f, 2.5f, 0.5f, 0.5f, 0.5f, 1.5f},
                      std::vector<float> {0.5f, 1.5f, 0.5f, 2.5f, 1.5f, 0.5f, 3.5f, 2.5f, 0.5f, 0.5f, 2.5f, 0.5f, 0.5f, 0.5f, 1.5f}),
        ConvertParams(ngraph::PartialShape {1, 1, 3, 5}, ngraph::element::f16, ngraph::element::f32,
                      std::vector<float16> {0.5f, 1.5f, 0.5f, 2.5f, 1.5f, 0.5f, 3.5f, 2.5f, 0.5f, 0.5f, 2.5f, 0.5f, 0.5f, 0.5f, 1.5f},
                      std::vector<float> {0.5f, 1.5f, 0.5f, 2.5f, 1.5f, 0.5f, 3.5f, 2.5f, 0.5f, 0.5f, 2.5f, 0.5f, 0.5f, 0.5f, 1.5f}),
        ConvertParams(ngraph::PartialShape {1, 1, 3, 5}, ngraph::element::f32, ngraph::element::f32,
                      std::vector<float> {0.5f, 1.5f, 0.5f, 2.5f, 1.5f, 0.5f, 3.5f, 2.5f, 0.5f, 0.5f, 2.5f, 0.5f, 0.5f, 0.5f, 1.5f},
                      std::vector<float> {0.5f, 1.5f, 0.5f, 2.5f, 1.5f, 0.5f, 3.5f, 2.5f, 0.5f, 0.5f, 2.5f, 0.5f, 0.5f, 0.5f, 1.5f}),

        // destination i4
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u1, ngraph::element::i4, std::vector<uint8_t> {0xA0}, std::vector<uint8_t> {0x10, 0x10}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u4, ngraph::element::i4, std::vector<uint8_t> {0x12, 0x03}, std::vector<uint8_t> {0x12, 0x03},
                      4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u8, ngraph::element::i4, std::vector<uint8_t> {1, 2, 0, 3}, std::vector<uint8_t> {0x12, 0x03},
                      4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u16, ngraph::element::i4, std::vector<uint16_t> {1, 2, 0, 3},
                      std::vector<uint8_t> {0x12, 0x03}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u32, ngraph::element::i4, std::vector<uint32_t> {1, 2, 0, 3},
                      std::vector<uint8_t> {0x12, 0x03}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u64, ngraph::element::i4, std::vector<uint64_t> {1, 2, 0, 3},
                      std::vector<uint8_t> {0x12, 0x03}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i4, ngraph::element::i4, std::vector<uint8_t> {0xFE, 0x03}, std::vector<uint8_t> {0xFE, 0x03},
                      4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i8, ngraph::element::i4, std::vector<int8_t> {-1, -2, 2, 3}, std::vector<uint8_t> {0xFE, 0x23},
                      4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i16, ngraph::element::i4, std::vector<int16_t> {-1, -2, 2, 3},
                      std::vector<uint8_t> {0xFE, 0x23}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i32, ngraph::element::i4, std::vector<int32_t> {-1, -2, 2, 3},
                      std::vector<uint8_t> {0xFE, 0x23}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i64, ngraph::element::i4, std::vector<int64_t> {-1, -2, 2, 3},
                      std::vector<uint8_t> {0xFE, 0x23}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f16, ngraph::element::i4, std::vector<ngraph::float16> {-1, -2, 0, 3},
                      std::vector<uint8_t> {0xFE, 0x03}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::bf16, ngraph::element::i4, std::vector<ngraph::bfloat16> {-1, -2, 0, 3},
                      std::vector<uint8_t> {0xFE, 0x03}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f32, ngraph::element::i4, std::vector<float> {-1, -2, 2, 3}, std::vector<uint8_t> {0xFE, 0x23},
                      4, 4),
        // destination i8
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::u1, ngraph::element::i8, std::vector<uint8_t> {0x81},
                      std::vector<int8_t> {1, 0, 0, 0, 0, 0, 0, 1}, 8),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u4, ngraph::element::i8, std::vector<uint8_t> {0x21, 0x43}, std::vector<int8_t> {2, 1, 4, 3},
                      4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u8, ngraph::element::i8, std::vector<uint8_t> {1, 2, 0, 3}, std::vector<int8_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u16, ngraph::element::i8, std::vector<uint16_t> {1, 2, 0, 3},
                      std::vector<int8_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u32, ngraph::element::i8, std::vector<uint32_t> {1, 2, 0, 3},
                      std::vector<int8_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u64, ngraph::element::i8, std::vector<uint64_t> {1, 2, 0, 3},
                      std::vector<int8_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i4, ngraph::element::i8, std::vector<uint8_t> {0x21, 0x43}, std::vector<int8_t> {2, 1, 4, 3},
                      4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i8, ngraph::element::i8, std::vector<int8_t> {-1, -2, 2, 3},
                      std::vector<int8_t> {-1, -2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i16, ngraph::element::i8, std::vector<int16_t> {-1, -2, 2, 3},
                      std::vector<int8_t> {-1, -2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i32, ngraph::element::i8, std::vector<int32_t> {-1, -2, 2, 3},
                      std::vector<int8_t> {-1, -2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i64, ngraph::element::i8, std::vector<int64_t> {-1, -2, 2, 3},
                      std::vector<int8_t> {-1, -2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f16, ngraph::element::i8, std::vector<ngraph::float16> {-1, -2, 0, 3},
                      std::vector<int8_t> {-1, -2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::bf16, ngraph::element::i8, std::vector<ngraph::bfloat16> {-1, -2, 0, 3},
                      std::vector<int8_t> {-1, -2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f32, ngraph::element::i8, std::vector<float> {-1, -2, 2, 3},
                      std::vector<int8_t> {-1, -2, 2, 3}),
        // destination i16
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::u1, ngraph::element::i16, std::vector<uint8_t> {0x81},
                      std::vector<int16_t> {1, 0, 0, 0, 0, 0, 0, 1}, 8),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u4, ngraph::element::i16, std::vector<uint8_t> {0x21, 0x43}, std::vector<int16_t> {2, 1, 4, 3},
                      4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u8, ngraph::element::i16, std::vector<uint8_t> {1, 2, 0, 3},
                      std::vector<int16_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u16, ngraph::element::i16, std::vector<uint16_t> {1, 2, 0, 3},
                      std::vector<int16_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u32, ngraph::element::i16, std::vector<uint32_t> {1, 2, 0, 3},
                      std::vector<int16_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u64, ngraph::element::i16, std::vector<uint64_t> {1, 2, 0, 3},
                      std::vector<int16_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i4, ngraph::element::i16, std::vector<uint8_t> {0x21, 0x43}, std::vector<int16_t> {2, 1, 4, 3},
                      4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i8, ngraph::element::i16, std::vector<int8_t> {-1, -2, 2, 3},
                      std::vector<int16_t> {-1, -2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i16, ngraph::element::i16, std::vector<int16_t> {-1, -2, 2, 3},
                      std::vector<int16_t> {-1, -2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i32, ngraph::element::i16, std::vector<int32_t> {-1, -2, 2, 3},
                      std::vector<int16_t> {-1, -2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i64, ngraph::element::i16, std::vector<int64_t> {-1, -2, 2, 3},
                      std::vector<int16_t> {-1, -2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f16, ngraph::element::i16, std::vector<ngraph::float16> {-1, -2, 0, 3},
                      std::vector<int16_t> {-1, -2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::bf16, ngraph::element::i16, std::vector<ngraph::bfloat16> {-1, -2, 0, 3},
                      std::vector<int16_t> {-1, -2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f32, ngraph::element::i16, std::vector<float> {-1, -2, 2, 3},
                      std::vector<int16_t> {-1, -2, 2, 3}),
        // destination i32
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::u1, ngraph::element::i32, std::vector<uint8_t> {0x81},
                      std::vector<int32_t> {1, 0, 0, 0, 0, 0, 0, 1}, 8),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u4, ngraph::element::i32, std::vector<uint8_t> {0x21, 0x43}, std::vector<int32_t> {2, 1, 4, 3},
                      4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u8, ngraph::element::i32, std::vector<uint8_t> {1, 2, 0, 3},
                      std::vector<int32_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u16, ngraph::element::i32, std::vector<uint16_t> {1, 2, 0, 3},
                      std::vector<int32_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u32, ngraph::element::i32, std::vector<uint32_t> {1, 2, 0, 3},
                      std::vector<int32_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u64, ngraph::element::i32, std::vector<uint64_t> {1, 2, 0, 3},
                      std::vector<int32_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i4, ngraph::element::i32, std::vector<uint8_t> {0x21, 0x43}, std::vector<int32_t> {2, 1, 4, 3},
                      4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i8, ngraph::element::i32, std::vector<int8_t> {-1, -2, 2, 3},
                      std::vector<int32_t> {-1, -2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i16, ngraph::element::i32, std::vector<int16_t> {-1, -2, 2, 3},
                      std::vector<int32_t> {-1, -2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i32, ngraph::element::i32, std::vector<int32_t> {-1, -2, 2, 3},
                      std::vector<int32_t> {-1, -2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i64, ngraph::element::i32, std::vector<int64_t> {-1, -2, 2, 3},
                      std::vector<int32_t> {-1, -2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f16, ngraph::element::i32, std::vector<ngraph::float16> {-1, -2, 0, 3},
                      std::vector<int32_t> {-1, -2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::bf16, ngraph::element::i32, std::vector<ngraph::bfloat16> {-1, -2, 0, 3},
                      std::vector<int32_t> {-1, -2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f32, ngraph::element::i32, std::vector<float> {-1, -2, 2, 3},
                      std::vector<int32_t> {-1, -2, 2, 3}),
        // destination i64
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::u1, ngraph::element::i64, std::vector<uint8_t> {0x81},
                      std::vector<int64_t> {1, 0, 0, 0, 0, 0, 0, 1}, 8),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u4, ngraph::element::i64, std::vector<uint8_t> {0x21, 0x43}, std::vector<int64_t> {2, 1, 4, 3},
                      4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u8, ngraph::element::i64, std::vector<uint8_t> {1, 2, 0, 3},
                      std::vector<int64_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u16, ngraph::element::i64, std::vector<uint16_t> {1, 2, 0, 3},
                      std::vector<int64_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u32, ngraph::element::i64, std::vector<uint32_t> {1, 2, 0, 3},
                      std::vector<int64_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u64, ngraph::element::i64, std::vector<uint64_t> {1, 2, 0, 3},
                      std::vector<int64_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i4, ngraph::element::i64, std::vector<uint8_t> {0x21, 0x43}, std::vector<int64_t> {2, 1, 4, 3},
                      4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i8, ngraph::element::i64, std::vector<int8_t> {-1, -2, 2, 3},
                      std::vector<int64_t> {-1, -2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i16, ngraph::element::i64, std::vector<int16_t> {-1, -2, 2, 3},
                      std::vector<int64_t> {-1, -2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i32, ngraph::element::i64, std::vector<int32_t> {-1, -2, 2, 3},
                      std::vector<int64_t> {-1, -2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i64, ngraph::element::i64, std::vector<int64_t> {-1, -2, 2, 3},
                      std::vector<int64_t> {-1, -2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f16, ngraph::element::i64, std::vector<ngraph::float16> {-1, -2, 0, 3},
                      std::vector<int64_t> {-1, -2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::bf16, ngraph::element::i64, std::vector<ngraph::bfloat16> {-1, -2, 0, 3},
                      std::vector<int64_t> {-1, -2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f32, ngraph::element::i64, std::vector<float> {-1, -2, 2, 3},
                      std::vector<int64_t> {-1, -2, 2, 3}),

        // destination u1
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::u1, ngraph::element::u1, std::vector<uint8_t> {0xA0}, std::vector<uint8_t> {0xA0}, 8, 8),
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::u4, ngraph::element::u1, std::vector<uint8_t> {0x10, 0x01, 0x00, 0x00},
                      std::vector<uint8_t> {0x90}, 8, 8),
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::u8, ngraph::element::u1, std::vector<uint8_t> {1, 0, 1, 0, 0, 0, 0, 1},
                      std::vector<uint8_t> {0xA1}, 8, 8),
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::u16, ngraph::element::u1, std::vector<uint16_t> {1, 0, 1, 0, 0, 0, 0, 1},
                      std::vector<uint8_t> {0xA1}, 8, 8),
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::u32, ngraph::element::u1, std::vector<uint32_t> {1, 0, 1, 0, 0, 0, 0, 1},
                      std::vector<uint8_t> {0xA1}, 8, 8),
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::u64, ngraph::element::u1, std::vector<uint64_t> {1, 0, 1, 0, 0, 0, 0, 1},
                      std::vector<uint8_t> {0xA1}, 8, 8),
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::i4, ngraph::element::u1, std::vector<uint8_t> {0x10, 0x01, 0x00, 0x00},
                      std::vector<uint8_t> {0x90}, 8, 8),
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::i8, ngraph::element::u1, std::vector<int8_t> {1, 0, 1, 0, 0, 0, 0, 1},
                      std::vector<uint8_t> {0xA1}, 8, 8),
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::i16, ngraph::element::u1, std::vector<int16_t> {1, 0, 1, 0, 0, 0, 0, 1},
                      std::vector<uint8_t> {0xA1}, 8, 8),
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::i32, ngraph::element::u1, std::vector<int32_t> {1, 0, 1, 0, 0, 0, 0, 1},
                      std::vector<uint8_t> {0xA1}, 8, 8),
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::i64, ngraph::element::u1, std::vector<int64_t> {1, 0, 1, 0, 0, 0, 0, 1},
                      std::vector<uint8_t> {0xA1}, 8, 8),
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::f16, ngraph::element::u1, std::vector<ngraph::float16> {1, 0, 1, 0, 0, 0, 0, 1},
                      std::vector<uint8_t> {0xA1}, 8, 8),
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::bf16, ngraph::element::u1, std::vector<ngraph::bfloat16> {1, 0, 1, 0, 0, 0, 0, 1},
                      std::vector<uint8_t> {0xA1}, 8, 8),
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::f32, ngraph::element::u1, std::vector<float> {1, 0, 1, 0, 0, 0, 0, 1},
                      std::vector<uint8_t> {0xA1}, 8, 8),

        // destination u4
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u1, ngraph::element::u4, std::vector<uint8_t> {0xA0}, std::vector<uint8_t> {0x10, 0x10}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u4, ngraph::element::u4, std::vector<uint8_t> {0x12, 0x03}, std::vector<uint8_t> {0x12, 0x03},
                      4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u8, ngraph::element::u4, std::vector<uint8_t> {1, 2, 0, 3}, std::vector<uint8_t> {0x12, 0x03},
                      4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u16, ngraph::element::u4, std::vector<uint16_t> {1, 2, 0, 3},
                      std::vector<uint8_t> {0x12, 0x03}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u32, ngraph::element::u4, std::vector<uint32_t> {1, 2, 0, 3},
                      std::vector<uint8_t> {0x12, 0x03}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u64, ngraph::element::u4, std::vector<uint64_t> {1, 2, 0, 3},
                      std::vector<uint8_t> {0x12, 0x03}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i4, ngraph::element::u4, std::vector<uint8_t> {0xFE, 0x03}, std::vector<uint8_t> {0xFE, 0x03},
                      4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i8, ngraph::element::u4, std::vector<int8_t> {-1, -2, 2, 3}, std::vector<uint8_t> {0xFE, 0x23},
                      4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i16, ngraph::element::u4, std::vector<int16_t> {-1, -2, 2, 3},
                      std::vector<uint8_t> {0xFE, 0x23}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i32, ngraph::element::u4, std::vector<int32_t> {-1, -2, 2, 3},
                      std::vector<uint8_t> {0xFE, 0x23}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i64, ngraph::element::u4, std::vector<int64_t> {-1, -2, 2, 3},
                      std::vector<uint8_t> {0xFE, 0x23}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f16, ngraph::element::u4, std::vector<ngraph::float16> {-1, -2, 0, 3},
                      std::vector<uint8_t> {0xFE, 0x03}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::bf16, ngraph::element::u4, std::vector<ngraph::bfloat16> {-1, -2, 0, 3},
                      std::vector<uint8_t> {0xFE, 0x03}, 4, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f32, ngraph::element::u4, std::vector<float> {-1, -2, 2, 3}, std::vector<uint8_t> {0xFE, 0x23},
                      4, 4),

        // destination u8
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::u1, ngraph::element::u8, std::vector<uint8_t> {0x81},
                      std::vector<uint8_t> {1, 0, 0, 0, 0, 0, 0, 1}, 8),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u4, ngraph::element::u8, std::vector<uint8_t> {0x21, 0x43}, std::vector<uint8_t> {2, 1, 4, 3},
                      4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u8, ngraph::element::u8, std::vector<uint8_t> {1, 2, 0, 3}, std::vector<uint8_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u16, ngraph::element::u8, std::vector<uint16_t> {1, 2, 0, 3},
                      std::vector<uint8_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u32, ngraph::element::u8, std::vector<uint32_t> {1, 2, 0, 3},
                      std::vector<uint8_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u64, ngraph::element::u8, std::vector<uint64_t> {1, 2, 0, 3},
                      std::vector<uint8_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i4, ngraph::element::u8, std::vector<uint8_t> {0x21, 0x43}, std::vector<uint8_t> {2, 1, 4, 3},
                      4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i8, ngraph::element::u8, std::vector<int8_t> {1, 2, 2, 3}, std::vector<uint8_t> {1, 2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i16, ngraph::element::u8, std::vector<int16_t> {1, 2, 2, 3},
                      std::vector<uint8_t> {1, 2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i32, ngraph::element::u8, std::vector<int32_t> {1, 2, 2, 3},
                      std::vector<uint8_t> {1, 2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i64, ngraph::element::u8, std::vector<int64_t> {1, 2, 2, 3},
                      std::vector<uint8_t> {1, 2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f16, ngraph::element::u8, std::vector<ngraph::float16> {1, 2, 0, 3},
                      std::vector<uint8_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::bf16, ngraph::element::u8, std::vector<ngraph::bfloat16> {1, 2, 0, 3},
                      std::vector<uint8_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f32, ngraph::element::u8, std::vector<float> {1, 2, 2, 3}, std::vector<uint8_t> {1, 2, 2, 3}),

        // destination u16
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::u1, ngraph::element::u16, std::vector<uint8_t> {0x81},
                      std::vector<uint16_t> {1, 0, 0, 0, 0, 0, 0, 1}, 8),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u4, ngraph::element::u16, std::vector<uint8_t> {0x21, 0x43},
                      std::vector<uint16_t> {2, 1, 4, 3}, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u8, ngraph::element::u16, std::vector<uint8_t> {1, 2, 0, 3},
                      std::vector<uint16_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u16, ngraph::element::u16, std::vector<uint16_t> {1, 2, 0, 3},
                      std::vector<uint16_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u32, ngraph::element::u16, std::vector<uint32_t> {1, 2, 0, 3},
                      std::vector<uint16_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u64, ngraph::element::u16, std::vector<uint64_t> {1, 2, 0, 3},
                      std::vector<uint16_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i4, ngraph::element::u16, std::vector<uint8_t> {0x21, 0x43},
                      std::vector<uint16_t> {2, 1, 4, 3}, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i8, ngraph::element::u16, std::vector<int8_t> {1, 2, 2, 3},
                      std::vector<uint16_t> {1, 2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i16, ngraph::element::u16, std::vector<int16_t> {1, 2, 2, 3},
                      std::vector<uint16_t> {1, 2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i32, ngraph::element::u16, std::vector<int32_t> {1, 2, 2, 3},
                      std::vector<uint16_t> {1, 2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i64, ngraph::element::u16, std::vector<int64_t> {1, 2, 2, 3},
                      std::vector<uint16_t> {1, 2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f16, ngraph::element::u16, std::vector<ngraph::float16> {1, 2, 0, 3},
                      std::vector<uint16_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::bf16, ngraph::element::u16, std::vector<ngraph::bfloat16> {1, 2, 0, 3},
                      std::vector<uint16_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f32, ngraph::element::u16, std::vector<float> {1, 2, 2, 3},
                      std::vector<uint16_t> {1, 2, 2, 3}),

        // destination u32
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::u1, ngraph::element::u32, std::vector<uint8_t> {0x81},
                      std::vector<uint32_t> {1, 0, 0, 0, 0, 0, 0, 1}, 8),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u4, ngraph::element::u32, std::vector<uint8_t> {0x21, 0x43},
                      std::vector<uint32_t> {2, 1, 4, 3}, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u8, ngraph::element::u32, std::vector<uint8_t> {1, 2, 0, 3},
                      std::vector<uint32_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u16, ngraph::element::u32, std::vector<uint16_t> {1, 2, 0, 3},
                      std::vector<uint32_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u32, ngraph::element::u32, std::vector<uint32_t> {1, 2, 0, 3},
                      std::vector<uint32_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u64, ngraph::element::u32, std::vector<uint64_t> {1, 2, 0, 3},
                      std::vector<uint32_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i4, ngraph::element::u32, std::vector<uint8_t> {0x21, 0x43},
                      std::vector<uint32_t> {2, 1, 4, 3}, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i8, ngraph::element::u32, std::vector<int8_t> {1, 2, 2, 3},
                      std::vector<uint32_t> {1, 2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i16, ngraph::element::u32, std::vector<int16_t> {1, 2, 2, 3},
                      std::vector<uint32_t> {1, 2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i32, ngraph::element::u32, std::vector<int32_t> {1, 2, 2, 3},
                      std::vector<uint32_t> {1, 2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i64, ngraph::element::u32, std::vector<int64_t> {1, 2, 2, 3},
                      std::vector<uint32_t> {1, 2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f16, ngraph::element::u32, std::vector<ngraph::float16> {1, 2, 0, 3},
                      std::vector<uint32_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::bf16, ngraph::element::u32, std::vector<ngraph::bfloat16> {1, 2, 0, 3},
                      std::vector<uint32_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f32, ngraph::element::u32, std::vector<float> {1, 2, 2, 3},
                      std::vector<uint32_t> {1, 2, 2, 3}),

        // destination u64
        ConvertParams(ngraph::PartialShape {8}, ngraph::element::u1, ngraph::element::u64, std::vector<uint8_t> {0x81},
                      std::vector<uint64_t> {1, 0, 0, 0, 0, 0, 0, 1}, 8),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u4, ngraph::element::u64, std::vector<uint8_t> {0x21, 0x43},
                      std::vector<uint64_t> {2, 1, 4, 3}, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u8, ngraph::element::u64, std::vector<uint8_t> {1, 2, 0, 3},
                      std::vector<uint64_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u16, ngraph::element::u64, std::vector<uint16_t> {1, 2, 0, 3},
                      std::vector<uint64_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u32, ngraph::element::u64, std::vector<uint32_t> {1, 2, 0, 3},
                      std::vector<uint64_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::u64, ngraph::element::u64, std::vector<uint64_t> {1, 2, 0, 3},
                      std::vector<uint64_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i4, ngraph::element::u64, std::vector<uint8_t> {0x21, 0x43},
                      std::vector<uint64_t> {2, 1, 4, 3}, 4),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i8, ngraph::element::u64, std::vector<int8_t> {1, 2, 2, 3},
                      std::vector<uint64_t> {1, 2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i16, ngraph::element::u64, std::vector<int16_t> {1, 2, 2, 3},
                      std::vector<uint64_t> {1, 2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i32, ngraph::element::u64, std::vector<int32_t> {1, 2, 2, 3},
                      std::vector<uint64_t> {1, 2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::i64, ngraph::element::u64, std::vector<int64_t> {1, 2, 2, 3},
                      std::vector<uint64_t> {1, 2, 2, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f16, ngraph::element::u64, std::vector<ngraph::float16> {1, 2, 0, 3},
                      std::vector<uint64_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::bf16, ngraph::element::u64, std::vector<ngraph::bfloat16> {1, 2, 0, 3},
                      std::vector<uint64_t> {1, 2, 0, 3}),
        ConvertParams(ngraph::PartialShape {4}, ngraph::element::f32, ngraph::element::u64, std::vector<float> {1, 2, 2, 3},
                      std::vector<uint64_t> {1, 2, 2, 3})));
