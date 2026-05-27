from collections.abc import Sequence
from typing import Any

import jpype

from neuralogic.core.constructs.function.function import AggregationFunction


class SoftmaxAggregation(AggregationFunction):
    """
    Represents a Softmax aggregation function.
    It can be parametrized by specific terms (variables) to aggregate over.
    """

    __slots__ = ("agg_terms", "var_terms")

    def __init__(
        self,
        name: str,
        *,
        agg_terms: Sequence[str] | None = None,
    ):
        """
        Parameters
        ----------
        name : str
            The name of the aggregation function.
        agg_terms : Sequence[str], optional
            The terms (variables) to aggregate over. Default: None.
        """
        super().__init__(name)
        self.term_indices = agg_terms
        self.agg_terms = agg_terms

    def __call__(self, *, agg_terms: Sequence[int] | None = None) -> Any:
        """
        Creates a new SoftmaxAggregation instance with the provided aggregation terms.

        Parameters
        ----------
        agg_terms : Sequence[int], optional
            The indices or names of terms to aggregate over. Default: None.

        Returns
        -------
        AggregationFunction
            The new SoftmaxAggregation instance.
        """
        softmax = SoftmaxAggregation(self.name, agg_terms=agg_terms)
        return AggregationFunction.__call__(softmax)

    def is_parametrized(self) -> bool:
        return self.agg_terms is not None

    def get(self) -> Any:
        return jpype.JClass("cz.cvut.fel.ida.algebra.functions.combination.Softmax")(self.term_indices)

    def __str__(self) -> str:
        if self.agg_terms is None:
            return "softmax"
        return f"softmax(agg_terms=[{', '.join(self.agg_terms)}])"

    def rule_head_dependant(self) -> bool:
        return self.agg_terms is not None

    def process_head(self, head) -> "SoftmaxAggregation":
        """
        Processes the rule head to determine the indices of the aggregation terms.

        Parameters
        ----------
        head : Any
            The rule head.

        Returns
        -------
        SoftmaxAggregation
            A new instance with the determined term indices.
        """
        term_indices = []

        for agg_term in set(self.agg_terms):
            if not agg_term[0].isupper():
                raise NotImplementedError(f"Softmax aggregable terms can be only variables. Provided: {agg_term}")

            for i, term in enumerate(head.terms):
                if agg_term == term:
                    term_indices.append(i)
                    break

        aggregation = SoftmaxAggregation(self.name, agg_terms=self.agg_terms)
        aggregation.term_indices = term_indices

        return aggregation
