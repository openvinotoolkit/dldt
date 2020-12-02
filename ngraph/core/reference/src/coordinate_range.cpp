//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include "ngraph/coordinate_range.hpp"

#include <cassert>
#include <numeric>
#include <stdexcept>

#include "ngraph/coordinate_index.hpp"

namespace ngraph
{
    namespace coordinates
    {
        namespace impl
        {
            namespace
            {
                std::vector<ssize_t> memory_strides(const Shape& shape)
                {
                    std::vector<ssize_t> mem_strides(shape.size(), 1);

                    if (shape.size() > 1)
                    {
                        for (auto i = shape.size() - 1; i-- > 0;)
                        {
                            mem_strides[i] = mem_strides[i + 1] * shape[i + 1];
                        }
                    }

                    return mem_strides;
                }

            } // namespace

            SliceRange::SliceRange(const Shape& source_shape,
                                   const Coordinate& source_start_corner,
                                   const Coordinate& source_end_corner,
                                   const Strides& source_strides)
                : m_source_shape{source_shape}
                , m_bounds{source_start_corner, source_end_corner}
                , m_source_strides{source_strides}
                , m_memory_strides(memory_strides(source_shape))
                , m_coordinate{source_start_corner}
                , m_index(coordinate_index(source_start_corner, source_shape))
            {
                const auto axis = m_source_shape.size();

                if (axis != m_bounds.m_lower.size())
                {
                    throw std::domain_error(
                        "Source start corner does not have the same number of axis as the "
                        "source "
                        "space "
                        "shape");
                }
                if (axis != m_bounds.m_upper.size())
                {
                    throw std::domain_error(
                        "Source end corner does not have the same number of axis as the source "
                        "space "
                        "shape");
                }
                if (axis != m_source_strides.size())
                {
                    throw std::domain_error(
                        "Source strides do not have the same number of axis as the source "
                        "space "
                        "shape");
                }
                if (axis != m_memory_strides.size())
                {
                    throw std::runtime_error("Something goes wrong");
                }
            }

            ssize_t SliceRange::copy_range_first_index() const { return m_index; }
            bool SliceRange::increment()
            {
                if (m_coordinate.empty())
                {
                    return false;
                }
                // omit last dim - it will be return in slice_range
                for (auto axis = m_coordinate.size() - 1; axis-- > 0;)
                {
                    const auto index_step = m_source_strides[axis] * m_memory_strides[axis];
                    m_coordinate[axis] += m_source_strides[axis];
                    m_index += index_step;
                    if (m_coordinate[axis] < m_bounds.m_upper[axis])
                    {
                        assert(m_index < static_cast<ptrdiff_t>(shape_size(m_source_shape)));
                        return true;
                    }
                    const auto difference = m_coordinate[axis] - m_bounds.m_lower[axis];
                    m_coordinate[axis] = m_bounds.m_lower[axis];

                    // back on beginning of axis memory
                    m_index -= difference * m_memory_strides[axis];
                }

                return false;
            }

            namespace
            {
                Coordinate start_coordinate(const Shape& s,
                                            const std::vector<signed char>& direction)
                {
                    Coordinate coordiante(s.size(), 0);
                    for (size_t i = 0; i < s.size(); ++i)
                    {
                        if (direction[i] < 0)
                        {
                            coordiante[i] = s[i] - 1;
                        }
                    }
                    return coordiante;
                }

                std::vector<signed char> axis_direcions(size_t size, const AxisSet& reversed_axis)
                {
                    const auto max_reversed_axis = [&] {
                        return *std::max_element(reversed_axis.begin(), reversed_axis.end());
                    };
                    if (!reversed_axis.empty() && !(max_reversed_axis() < size))
                    {
                        throw std::domain_error(
                            "Reversed axis have axes above the source space shape");
                    }

                    std::vector<signed char> directions(size, 1);
                    for (auto i : reversed_axis)
                    {
                        directions[i] = -1;
                    }
                    return directions;
                }
            } // namespace

            ReverseRange::ReverseRange(const Shape& source_shape, const AxisSet& reversed_axis)
                : m_source_shape{source_shape}
                , m_memory_strides(memory_strides(source_shape))
                , m_axis_directions(axis_direcions(source_shape.size(), reversed_axis))
                , m_coordinate(source_shape.size(), 0)
                , m_index(coordinate_index(start_coordinate(source_shape, m_axis_directions),
                                           source_shape))
            {
            }

            ReverseRange::value_type ReverseRange::get_value() const
            {
                const ssize_t end_index = m_index + last_dim_size();

                const ssize_t step =
                    static_cast<ssize_t>(m_memory_strides.back()) * m_axis_directions.back();

                return Range{m_index, end_index, step};
            }

            bool ReverseRange::increment()
            {
                if (m_coordinate.empty())
                {
                    return false;
                }
                // omit last dim - it will be return in reverse_range
                for (auto axis = m_coordinate.size() - 1; axis-- > 0;)
                {
                    const auto index_step = m_memory_strides[axis];
                    ++m_coordinate[axis];
                    if (m_axis_directions[axis] > 0)
                    {
                        m_index += index_step;
                    }
                    else
                    {
                        m_index -= index_step;
                    }
                    if (m_coordinate[axis] < m_source_shape[axis])
                    {
                        assert(0 <= m_index && m_index < shape_size(m_source_shape));
                        return true;
                    }
                    m_coordinate[axis] = 0;

                    // back on beginning of axis memory
                    if (m_axis_directions[axis] > 0)
                    {
                        m_index -= m_source_shape[axis] * m_memory_strides[axis];
                    }
                    else
                    {
                        m_index += m_source_shape[axis] * m_memory_strides[axis];
                    }
                }
                return false;
            }

            ssize_t ReverseRange::last_dim_size() const noexcept
            {
                const ssize_t dir = m_axis_directions.back() > 0 ? 1 : -1;
                return dir * static_cast<ssize_t>(m_source_shape.back());
            }

        } // namespace impl

    } // namespace coordinates
} // namespace ngraph
