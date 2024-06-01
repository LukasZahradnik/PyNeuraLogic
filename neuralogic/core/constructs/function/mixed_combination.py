import jpype

from neuralogic.core.constructs.function.function import Combination


class MixedCombination(Combination):
    __slots__ = ("combination_graph",)

    def __init__(
        self,
        name: str,
        *,
        combination_graph,
    ):
        super().__init__(name)
        self.combination_graph = combination_graph

    def __call__(self, entity=None, *, combination_graph):
        concat = MixedCombination(self.name, combination_graph=combination_graph)
        return Combination.__call__(concat, entity)

    def is_parametrized(self) -> bool:
        return True

    def get(self):
        return jpype.JClass("cz.cvut.fel.ida.algebra.functions.combination.MixedCombination")(
            self.name, self.combination_graph
        )

    def __str__(self):
        return self.name
