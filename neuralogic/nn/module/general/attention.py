import math

from neuralogic.core.constructs.function import Transformation, Combination
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
        terms = [f"X{i}" for i in range(1, self.arity)]
        qterms = [f"X{i}" for i in range(self.arity)]

        d_k = 1 / math.sqrt(self.embed_dim)

        dk_rel = R.get(f"{self.output_name}__dk")
        dot_rel = R.get(f"{self.output_name}__dot")

        return [
            dk_rel[d_k].fixed(),
            (dot_rel(qterms) <= (dk_rel, R.get(self.key_name)(terms), R.get(self.query_name)(qterms)))
            | [Combination.PRODUCT, Transformation.IDENTITY],
            dot_rel / self.arity | [Transformation.IDENTITY],
            (
                R.get(self.output_name)(qterms)
                <= (R.get(self.value_name)(terms).T, Transformation.SOFTMAX(dot_rel(qterms)))
            )
            | [Transformation.IDENTITY, Combination.PRODUCT],
            R.get(self.output_name) / self.arity | [Transformation.IDENTITY],
        ]