from typing import Dict

import jpype

from neuralogic.core.constructs.function.concat import ConcatComb, Concat
from neuralogic.core.constructs.function.function import Transformation, Combination, Aggregation, Function
from neuralogic.core.constructs.function.reshape import Reshape
from neuralogic.core.constructs.function.mixed_combination import MixedCombination
from neuralogic.core.constructs.function.slice import Slice
from neuralogic.core.constructs.function.softmax import Softmax

_special_namings = {"LEAKY_RELU": "LEAKYRELU", "TRANSP": "TRANSPOSE"}

for function_name in Transformation.__annotations__:
    setattr(Transformation, function_name, Transformation(_special_namings.get(function_name, function_name)))


Transformation.SLICE = Slice("SLICE")
Transformation.RESHAPE = Reshape("RESHAPE")


for function_name in Combination.__annotations__:
    setattr(Combination, function_name, Combination(function_name))


Combination.CONCAT = ConcatComb("CONCAT")


for function_name in Aggregation.__annotations__:
    setattr(Aggregation, function_name, Aggregation(function_name))


Aggregation.CONCAT = Concat("CONCAT")
Aggregation.SOFTMAX = Softmax("SOFTMAX")


class CombinationWrap:
    __slots__ = "left", "right", "combination"

    def __init__(self, left, right, combination: Combination):
        self.left = left
        self.right = right
        self.combination = combination

    def __add__(self, other):
        return CombinationWrap(self, other, Combination.SUM)

    def __mul__(self, other):
        return CombinationWrap(self, other, Combination.ELPRODUCT)

    def __matmul__(self, other):
        return CombinationWrap(self, other, Combination.PRODUCT)

    def __str__(self):
        if not isinstance(self.left, CombinationWrap) and not isinstance(self.right, CombinationWrap):
            return f"{self.combination}"
        if not isinstance(self.left, CombinationWrap):
            return f"{self.combination}({self.right.to_str()})"
        if not isinstance(self.right, CombinationWrap):
            return f"{self.combination}({self.left.to_str()})"

        return f"{self.combination}({self.left.to_str()}, {self.right.to_str()})"

    def __iter__(self):
        if isinstance(self.left, CombinationWrap):
            for a in self.left:
                yield a
        if not isinstance(self.left, CombinationWrap):
            yield self.left

        if isinstance(self.right, CombinationWrap):
            for a in self.right:
                yield a
        if not isinstance(self.right, CombinationWrap):
            yield self.right

    def to_combination(self) -> Combination:
        combination_graph = self._get_combination_node({}, 0)
        return MixedCombination(name=self.to_str(), combination_graph=combination_graph)

    def _get_combination_node(self, input_counter: Dict[int, int], start_index: int = 0):
        left_node = None
        right_node = None

        left_index = -1
        right_index = -1

        if isinstance(self.left, CombinationWrap):
            left_node = self.left._get_combination_node(input_counter)
        else:
            if id(self.left) not in input_counter:
                input_counter[id(self.left)] = len(input_counter) + start_index
            left_index = input_counter[id(self.left)]

        if isinstance(self.right, CombinationWrap):
            right_node = self.right._get_combination_node(input_counter)
        else:
            if id(self.right) not in input_counter:
                input_counter[id(self.right)] = len(input_counter) + start_index
            right_index = input_counter[id(self.right)]

        class_name = "cz.cvut.fel.ida.algebra.functions.combination.MixedCombination.MixedCombinationNode"

        return jpype.JClass(class_name)(self.combination.get(), left_node, right_node, left_index, right_index)

    def to_str(self):
        return self.__str__()


__all__ = ["Transformation", "Combination", "Aggregation", "Function", "CombinationWrap"]
