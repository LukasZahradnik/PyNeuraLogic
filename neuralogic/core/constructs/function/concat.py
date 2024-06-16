import jpype

from neuralogic.core.constructs.function.function import AggregationFunction, CombinationFunction


class ConcatCombination(CombinationFunction):
    __slots__ = ("axis",)

    def __init__(
        self,
        name: str,
        *,
        axis: int = -1,
    ):
        super().__init__(name)
        self.axis = axis

    def __call__(self, *relations, axis: int = -1):
        concat = ConcatCombination(self.name, axis=axis)
        return CombinationFunction.__call__(concat, *relations)

    def is_parametrized(self) -> bool:
        return self.axis != -1

    def get(self):
        return jpype.JClass("cz.cvut.fel.ida.algebra.functions.combination.Concatenation")(self.axis)

    def __str__(self):
        if self.axis == -1:
            return "concat"
        return f"concat(axis={self.axis})"


class ConcatAggregation(AggregationFunction):
    __slots__ = ("axis",)

    def __init__(
        self,
        name: str,
        *,
        axis: int = -1,
    ):
        super().__init__(name)
        self.axis = axis

    def __call__(self, *, axis: int = -1):
        concat = ConcatAggregation(self.name, axis=axis)
        return AggregationFunction.__call__(concat)

    def is_parametrized(self) -> bool:
        return self.axis != -1

    def get(self):
        return jpype.JClass("cz.cvut.fel.ida.algebra.functions.combination.Concatenation")(self.axis)

    def __str__(self):
        if self.axis == -1:
            return "concat"
        return f"concat(axis={self.axis})"
