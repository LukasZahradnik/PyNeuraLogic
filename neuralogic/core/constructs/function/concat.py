import jpype

from neuralogic.core.constructs.function.function import Aggregation, Combination


class ConcatComb(Combination):
    __slots__ = ("axis",)

    def __init__(
        self,
        name: str,
        *,
        axis: int = -1,
    ):
        super().__init__(name)
        self.axis = axis

    def __call__(self, entity=None, *, axis: int = -1):
        concat = ConcatComb(self.name, axis=axis)
        return Combination.__call__(concat, entity)

    def is_parametrized(self) -> bool:
        return self.axis != -1

    def get(self):
        return jpype.JClass("cz.cvut.fel.ida.algebra.functions.combination.Concatenation")(self.axis)

    def __str__(self):
        if self.axis == -1:
            return "concat"
        return f"concat(axis={self.axis})"


class Concat(Aggregation):
    __slots__ = ("axis",)

    def __init__(
        self,
        name: str,
        *,
        axis: int = -1,
    ):
        super().__init__(name)
        self.axis = axis

    def __call__(self, entity=None, *, axis: int = -1):
        concat = Concat(self.name, axis=axis)
        return Aggregation.__call__(concat, entity)

    def is_parametrized(self) -> bool:
        return self.axis != -1

    def get(self):
        return jpype.JClass("cz.cvut.fel.ida.algebra.functions.combination.Concatenation")(self.axis)

    def __str__(self):
        if self.axis == -1:
            return "concat"
        return f"concat(axis={self.axis})"
