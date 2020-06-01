# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""Helper classes for aggregating TensorIterator input/output desciptor attributes."""

from typing import List

from ngraph.impl import Node
from ngraph.impl.op import Parameter


class GraphBody(object):
    """Class containing graph parameters and results."""

    def __init__(
        self,
        parameters: List[Parameter],
        results: List[Node],
    ) -> None:
        self.parameters = parameters
        self.results = results

    def serialize(self) -> dict:
        return {
            "parameters": self.parameters,
            "results": self.results,
        }


class TensorIteratorInputDesc(object):
    """Represents a generic input descriptor for TensorIterator operator."""

    def __init__(
        self,
        input_idx: int,
        body_parameter_idx: int,
    ) -> None:
        self.input_idx = input_idx
        self.body_parameter_idx = body_parameter_idx

    def serialize(self) -> dict:
        return {
            "input_idx": self.input_idx,
            "body_parameter_idx": self.body_parameter_idx,
        }


class TensorIteratorSliceInputDesc(TensorIteratorInputDesc):
    """Represents a TI graph body input formed from slicec of an TI input."""

    def __init__(
        self,
        input_idx: int,
        body_parameter_idx: int,
        start: int,
        stride: int,
        part_size: int,
        end: int,
        axis: int,
    ) -> None:
        super(TensorIteratorSliceInputDesc, self).__init__(input_idx, body_parameter_idx)
        self.start = start
        self.stride = stride
        self.part_size = part_size
        self.end = end
        self.axis = axis

    def serialize(self) -> dict:
        output = super(TensorIteratorSliceInputDesc, self).serialize()
        output["start"] = self.start
        output["stride"] = self.stride
        output["part_size"] = self.part_size
        output["end"] = self.end
        output["axis"] = self.axis
        return output


class TensorIteratorMergedInputDesc(TensorIteratorInputDesc):
    """Represents a TI graph body input with initial value in the first iteration.

    Later on, this input value is computed inside graph body.
    """

    def __init__(
        self,
        input_idx: int,
        body_parameter_idx: int,
        body_value_idx: int,
    ) -> None:
        super(TensorIteratorMergedInputDesc, self).__init__(input_idx, body_parameter_idx)
        self.body_value_idx = body_value_idx

    def serialize(self) -> dict:
        output = super(TensorIteratorMergedInputDesc, self).serialize()
        output["body_value_idx"] = self.body_value_idx
        return output


class TensorIteratorInvariantInputDesc(TensorIteratorInputDesc):
    """Represents a TI graph body input that has invariant value during iteration."""

    def __init__(
        self,
        input_idx: int,
        body_parameter_idx: int,
    ) -> None:
        super(TensorIteratorInvariantInputDesc, self).__init__(input_idx, body_parameter_idx)


class TensorIteratorOutputDesc(object):
    """Represents a generic output descriptor for TensorIterator operator."""

    def __init__(
        self,
        body_value_idx: int,
        output_idx: int,
    ) -> None:
        self.body_value_idx = body_value_idx
        self.output_idx = output_idx

    def serialize(self) -> dict:
        return {
            "body_value_idx": self.body_value_idx,
            "output_idx": self.output_idx,
        }


class TensorIteratorBodyOutputDesc(TensorIteratorOutputDesc):
    """Represents an output from a specific iteration."""

    def __init__(
        self,
        body_value_idx: int,
        output_idx: int,
        iteration: int,
    ) -> None:
        super(TensorIteratorBodyOutputDesc, self).__init__(body_value_idx, output_idx)
        self.iteration = iteration

    def serialize(self) -> dict:
        output = super(TensorIteratorBodyOutputDesc, self).serialize()
        output["iteration"] = self.iteration
        return output


class TensorIteratorConcatOutputDesc(TensorIteratorOutputDesc):
    """Represents an output produced by concatenation of output from each iteration."""

    def __init__(
        self,
        body_value_idx: int,
        output_idx: int,
        start: int,
        stride: int,
        part_size: int,
        end: int,
        axis: int,
    ) -> None:
        super(TensorIteratorConcatOutputDesc, self).__init__(body_value_idx, output_idx)
        self.start = start
        self.stride = stride
        self.part_size = part_size
        self.end = end
        self.axis = axis

    def serialize(self) -> dict:
        output = super(TensorIteratorConcatOutputDesc, self).serialize()
        output["start"] = self.start
        output["stride"] = self.stride
        output["part_size"] = self.part_size
        output["end"] = self.end
        output["axis"] = self.axis
        return output
