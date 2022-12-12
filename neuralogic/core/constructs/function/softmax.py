from typing import Sequence

import jpype

from neuralogic.core.constructs.function.function import Aggregation


class Softmax(Aggregation):
    __slots__ = ("agg_terms", "var_terms")

    def __init__(
        self,
        name: str,
        *,
        agg_terms: Sequence[str] = None,
    ):
        super().__init__(name)
        self.term_indices = agg_terms
        self.agg_terms = agg_terms

    def __call__(self, entity=None, *, agg_terms: Sequence[int] = None):
        softmax = Softmax(self.name, agg_terms=agg_terms)
        return Aggregation.__call__(softmax, entity)

    def is_parametrized(self) -> bool:
        return self.agg_terms is not None

    def get(self):
        return jpype.JClass("cz.cvut.fel.ida.algebra.functions.combination.Softmax")(self.term_indices)

    def __str__(self):
        if self.agg_terms is None:
            return "softmax"
        return f"softmax(agg_terms=[{', '.join(self.agg_terms)}])"

    def rule_head_dependant(self) -> bool:
        return self.agg_terms is not None

    def process_head(self, head) -> "Softmax":
        term_indices = []

        for agg_term in set(self.agg_terms):
            if not agg_term[0].isupper():
                raise NotImplementedError(f"Softmax aggregable terms can be only variables. Provided: {agg_term}")

            for i, term in enumerate(head.terms):
                if agg_term == term:
                    term_indices.append(i)
                    break

        aggregation = Softmax(self.name, agg_terms=self.agg_terms)
        aggregation.term_indices = term_indices

        return aggregation
