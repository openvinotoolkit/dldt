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

#pragma once

#include <cmath>
#include <cstddef>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void hard_sigmoid(const T* arg, const T* alpha, const T* beta, T* out, size_t count)
            {
                T alpha_value = alpha[0];
                T beta_value = beta[0];
                for (size_t i = 0; i < count; i++)
                {
                    out[i] =
                        std::max<T>(0.0f, std::min<T>(1.0f, alpha_value * arg[i] + beta_value));
                }
            }
        }
    }
}
