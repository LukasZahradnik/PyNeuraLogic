from typing import Dict

import jpype

from neuralogic.core.constructs.function.enum import Combination
from neuralogic.core.constructs.function.function_graph import FunctionGraph
from neuralogic.core.constructs.function.function import Function


class FContainer:
    __slots__ = "nodes", "function"

    def __init__(self, nodes, function: Function):
        self.function = function
        self.nodes = nodes if not self.function.can_flatten else self.get_flattened_nodes(nodes, function)

    @staticmethod
    def get_flattened_nodes(nodes, function: Function):
        new_nodes = []
        for node in nodes:
            if not isinstance(node, FContainer):
                new_nodes.append(node)
                continue

            if node.function.name == function.name:
                new_nodes.extend(node.nodes)
            else:
                new_nodes.append(node)
        return tuple(new_nodes)

    def __add__(self, other):
        return FContainer((self, other), Combination.SUM)

    def __mul__(self, other):
        return FContainer((self, other), Combination.ELPRODUCT)

    def __matmul__(self, other):
        return FContainer((self, other), Combination.PRODUCT)

    def __str__(self):
        if self.function.operator is not None:
            return f" {self.function.operator} ".join(
                node.to_str(True) if isinstance(node, FContainer) else node.to_str() for node in self.nodes
            )

        args = ", ".join(node.to_str() for node in self.nodes)

        if args:
            return f"{self.function}({args})"
        return f"{self.function}"

    @property
    def name(self):
        args = ", ".join(str(node.function) for node in self.nodes if isinstance(node, FContainer))

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
        return FunctionGraph(name=self.name, function_graph=graph)

    def _get_function_node(self, input_counter: Dict[int, int], start_index: int = 0):
        from neuralogic.core.constructs.relation import BaseRelation

        next_indices = [-1] * len(self.nodes)
        next_nodes = [None] * len(self.nodes)

        for i, node in enumerate(self.nodes):
            if isinstance(node, FContainer):
                next_node = node._get_function_node(input_counter)
                if next_node is None:
                    continue
                next_nodes[i] = next_node
            elif isinstance(node, BaseRelation):
                idx = id(node)

                if node.predicate.hidden or node.predicate.special or node.predicate.name.startswith("_"):
                    continue

                if idx not in input_counter:
                    input_counter[idx] = len(input_counter) + start_index
                next_indices[i] = input_counter[idx]
            else:
                raise ValueError(f"{node} of type {type(node)} inside of body function is not supported")

        filtered_next_node = []
        filtered_next_indices = []

        for i, (node, index) in enumerate(zip(next_nodes, next_indices)):
            if node is not None or index != -1:
                filtered_next_node.append(node)
                filtered_next_indices.append(index)

        if not filtered_next_node or not filtered_next_indices:
            return None

        class_name = "cz.cvut.fel.ida.algebra.functions.combination.FunctionGraph.FunctionGraphNode"

        return jpype.JClass(class_name)(self.function.get(), filtered_next_node, filtered_next_indices)

    def to_str(self, parentheses_wrap: bool = False):
        if parentheses_wrap and self.function.operator is not None:
            return f"({self})"
        return self.__str__()
