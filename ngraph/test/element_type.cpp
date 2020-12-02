//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <map>

#include "gtest/gtest.h"

#include "ngraph/type/element_type.hpp"

using namespace ngraph;

TEST(element_type, from)
{
    EXPECT_EQ(element::from<char>(),     element::Type_t::boolean);
    EXPECT_EQ(element::from<bool>(),     element::Type_t::boolean);
    EXPECT_EQ(element::from<float>(),    element::Type_t::f32);
    EXPECT_EQ(element::from<double>(),   element::Type_t::f64);
    EXPECT_EQ(element::from<int8_t>(),   element::Type_t::i8);
    EXPECT_EQ(element::from<int16_t>(),  element::Type_t::i16);
    EXPECT_EQ(element::from<int32_t>(),  element::Type_t::i32);
    EXPECT_EQ(element::from<int64_t>(),  element::Type_t::i64);
    EXPECT_EQ(element::from<uint8_t>(),  element::Type_t::u8);
    EXPECT_EQ(element::from<uint16_t>(), element::Type_t::u16);
    EXPECT_EQ(element::from<uint32_t>(), element::Type_t::u32);
    EXPECT_EQ(element::from<uint64_t>(), element::Type_t::u64);
}

TEST(element_type, mapable)
{
    std::map<element::Type, std::string> test_map;

    test_map.insert({element::Type_t::f32, "float"});
}

TEST(element_type, merge_both_dynamic)
{
    element::Type t;
    ASSERT_TRUE(element::Type::merge(t, element::Type_t::dynamic, element::Type_t::dynamic));
    ASSERT_TRUE(t.is_dynamic());
}

TEST(element_type, merge_left_dynamic)
{
    element::Type t;
    ASSERT_TRUE(element::Type::merge(t, element::Type_t::dynamic, element::Type_t::u64));
    ASSERT_TRUE(t.is_static());
    ASSERT_EQ(t, element::Type_t::u64);
}

TEST(element_type, merge_right_dynamic)
{
    element::Type t;
    ASSERT_TRUE(element::Type::merge(t, element::Type_t::i16, element::Type_t::dynamic));
    ASSERT_TRUE(t.is_static());
    ASSERT_EQ(t, element::Type_t::i16);
}

TEST(element_type, merge_both_static_equal)
{
    element::Type t;
    ASSERT_TRUE(element::Type::merge(t, element::Type_t::f64, element::Type_t::f64));
    ASSERT_TRUE(t.is_static());
    ASSERT_EQ(t, element::Type_t::f64);
}

TEST(element_type, merge_both_static_unequal)
{
    element::Type t = element::Type_t::f32;
    ASSERT_FALSE(element::Type::merge(t, element::Type_t::i8, element::Type_t::i16));
    ASSERT_TRUE(t.is_static());
    ASSERT_EQ(t, element::Type_t::f32);
}
