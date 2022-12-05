from typing import Sequence

import jpype

from neuralogic.core.constructs.function.function import Aggregation


class Softmax(Aggregation):
    __slots__ = ("agg_terms",)

    def __init__(
        self,
        name: str,
        *,
        agg_terms: Sequence[int] = None,
    ):
        super().__init__(name)
        if agg_terms is not None:
            agg_terms = tuple(int(i) for i in agg_terms)
        self.agg_terms = agg_terms

    def __call__(self, entity=None, *, agg_terms: Sequence[int] = None):
        softmax = Softmax(self.name, agg_terms=agg_terms)
        return Aggregation.__call__(softmax, entity)

    def is_parametrized(self) -> bool:
        return self.agg_terms is not None

    def get(self):
        return jpype.JClass("cz.cvut.fel.ida.algebra.functions.combination.Softmax")(self.agg_terms)

    def __str__(self):
        if self.agg_terms is None:
            return "softmax"
        return f"softmax(agg_terms={self.agg_terms})"
