from typing import Dict

import jpype

from neuralogic.core.constructs.function.function_graph import FunctionGraph
from neuralogic.core.constructs.function.function import Function, Combination


class FContainer:
    __slots__ = "nodes", "function"

    def __init__(self, nodes, function: Function):
        self.function = function
        self.nodes = nodes

    def __add__(self, other):
        return FContainer((self, other), Combination.SUM)

    def __mul__(self, other):
        return FContainer((self, other), Combination.ELPRODUCT)

    def __matmul__(self, other):
        return FContainer((self, other), Combination.PRODUCT)

    def __str__(self):
        args = ", ".join(node.to_str() for node in self.nodes if isinstance(node, FContainer))

        if args:
            return f"{self.function}({args})"
        return f"{self.function}"

    def __iter__(self):
        for node in self.nodes:
            if isinstance(node, FContainer):
                for a in node:
                    yield a
            else:
                yield node

    def to_function(self) -> Function:
        graph = self._get_function_node({}, 0)
        return FunctionGraph(name=self.to_str(), function_graph=graph)

    def _get_function_node(self, input_counter: Dict[int, int], start_index: int = 0):
        next_indices = [-1] * len(self.nodes)
        next_nodes = [None] * len(self.nodes)

        for i, node in enumerate(self.nodes):
            if isinstance(node, FContainer):
                next_nodes[i] = node._get_function_node(input_counter)
            else:
                idx = id(node)

                if idx not in input_counter:
                    input_counter[idx] = len(input_counter) + start_index
                next_indices[i] = input_counter[idx]

        class_name = "cz.cvut.fel.ida.algebra.functions.combination.FunctionGraph.FunctionGraphNode"

        return jpype.JClass(class_name)(self.function.get(), next_nodes, next_indices)

    def to_str(self):
        return self.__str__()
