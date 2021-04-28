// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ngraph
{
    namespace test
    {
        /// These templates should be specialized for each test engine and they should contain
        /// a "static constexpr const bool value" member set to true or false.
        /// These traits are used in engine_factory.hpp

        /// Indicates that a given Engine can be constructed for different devices (IE engines)
        template <typename Engine>
        struct supports_devices;

        /// Indicates that a given Engine supports dynamic shapes
        template <typename Engine>
        struct supports_dynamic;

        /// Indicates that a given Engine supports collecting statistic about ops used in test cases
        template <typename Engine>
        struct supports_ops_stats_collection;
        /// Example:
        ///
        // template <>
        // struct supports_dynamic<EngineName> {
        //     static constexpr const bool value = true;
        // };
    }
}
