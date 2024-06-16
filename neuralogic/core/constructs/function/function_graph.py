import jpype

from neuralogic.core.constructs.function.function import Function


class FunctionGraph(Function):
    __slots__ = ("function_graph",)

    def __init__(
        self,
        name: str,
        *,
        function_graph,
    ):
        super().__init__(name)
        self.function_graph = function_graph

    def is_parametrized(self) -> bool:
        return True

    def get(self):
        return jpype.JClass("cz.cvut.fel.ida.algebra.functions.combination.FunctionGraph")(
            self.name, self.function_graph
        )

    def __str__(self):
        return self.name
