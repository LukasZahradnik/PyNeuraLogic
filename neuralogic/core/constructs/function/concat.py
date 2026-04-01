import jpype

from neuralogic.core.constructs.function.function import AggregationFunction, CombinationFunction


class ConcatCombination(CombinationFunction):
    """
    Represents a concatenation combination function that joins multiple tensors along a specified axis.
    """
    __slots__ = ("axis",)

    def __init__(
        self,
        name: str,
        *,
        axis: int = -1,
    ):
        """
        Parameters
        ----------
        name : str
            The name of the function.
        axis : int, optional
            The axis along which to concatenate. Default: -1.
        """
        super().__init__(name)
        self.axis = axis

    def __call__(self, *relations, axis: int = -1):
        """
        Creates a new ConcatCombination instance with the provided axis and applies it to the relations.

        Parameters
        ----------
        relations : Any
            The relations to concatenate.
        axis : int, optional
            The axis to concatenate along. Default: -1.

        Returns
        -------
        CombinationFunction
            The new ConcatCombination instance (attached to the relations).
        """
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
    """
    Represents a concatenation aggregation function that joins multiple groundings along a specified axis.
    """
    __slots__ = ("axis",)

    def __init__(
        self,
        name: str,
        *,
        axis: int = -1,
    ):
        """
        Parameters
        ----------
        name : str
            The name of the function.
        axis : int, optional
            The axis along which to aggregate. Default: -1.
        """
        super().__init__(name)
        self.axis = axis

    def __call__(self, *, axis: int = -1):
        """
        Creates a new ConcatAggregation instance with the provided axis.

        Parameters
        ----------
        axis : int, optional
            The axis to aggregate along. Default: -1.

        Returns
        -------
        AggregationFunction
            The new ConcatAggregation instance.
        """
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
