import math

from neuralogic.core.constructs.function import Transformation, Combination, Aggregation
from neuralogic.core.constructs.factories import R
from neuralogic.nn.module.module import Module


class Attention(Module):
    def __init__(
        self,
        embed_dim: int,
        output_name: str,
        query_name: str,
        key_name: str,
        value_name: str,
        arity: int = 1,
    ):
        self.embed_dim = embed_dim
        self.output_name = output_name
        self.query_name = query_name
        self.key_name = key_name
        self.value_name = value_name
        self.arity = arity

    def __call__(self):
        terms = [f"X{i}" for i in range(self.arity)]
        val_terms = [f"X{i}" for i in range(self.arity - 1)]
        agg_terms = [*val_terms, f"X{self.arity}"]

        d_k = 1 / math.sqrt(self.embed_dim)

        dk_rel = R.get(f"{self.output_name}__dk")
        dot_rel = R.get(f"{self.output_name}__dot")
        values_rel = R.get(f"{self.output_name}__vals")

        return [
            dk_rel[d_k].fixed(),
            (dot_rel(terms) <= (dk_rel, R.get(self.key_name)(agg_terms).T, R.get(self.query_name)(terms)))
            | [Combination.PRODUCT, Transformation.IDENTITY, Aggregation.CONCAT],
            dot_rel / self.arity | [Transformation.SOFTMAX],
            (values_rel(val_terms) <= R.get(self.value_name)(agg_terms).T)
            | [Transformation.IDENTITY, Aggregation.CONCAT(axis=0)],
            values_rel / (self.arity - 1) | [Transformation.IDENTITY],
            (R.get(self.output_name)(terms) <= (dot_rel(terms).T, values_rel(val_terms)))
            | [Transformation.IDENTITY, Combination.PRODUCT],
            R.get(self.output_name) / self.arity | [Transformation.IDENTITY],
        ]
