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
        terms = [f"X{i}" for i in range(self.arity - 1)]

        k_terms = [*terms, "Y"]
        h_terms = [*terms, "X", "Y"]
        q_terms = [*terms, "X"]

        d_k = 1 / math.sqrt(self.embed_dim)

        dk_rel = R.get(f"{self.output_name}__dk")
        dot_rel = R.get(f"{self.output_name}__dot")

        metadata = [Combination.PRODUCT, Transformation.IDENTITY, Aggregation.SOFTMAX(agg_terms=[self.arity])]
        out_metadata = [Combination.PRODUCT, Aggregation.SUM, Transformation.IDENTITY]

        return [
            dk_rel[d_k].fixed(),
            (dot_rel(h_terms) <= (dk_rel, R.get(self.key_name)(k_terms).T, R.get(self.query_name)(q_terms))) | metadata,
            dot_rel / (self.arity + 1) | [Transformation.IDENTITY],
            (R.get(self.output_name)(q_terms) <= (dot_rel(h_terms), R.get(self.value_name)(k_terms))) | out_metadata,
            R.get(self.output_name) / self.arity | [Transformation.IDENTITY],
        ]


class MultiheadAttention(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        output_name: str,
        query_name: str,
        key_name: str,
        value_name: str,
        vdim: int = None,
        kdim: int = None,
        arity: int = 1,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_name = output_name
        self.query_name = query_name
        self.key_name = key_name
        self.value_name = value_name
        self.vdim = vdim if vdim is not None else embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.arity = arity

    def __call__(self):
        terms = [f"X{i}" for i in range(self.arity)]
        qproj_name = f"{self.output_name}__qproj"
        kproj_name = f"{self.output_name}__kproj"
        vproj_name = f"{self.output_name}__vproj"
        attention_name = f"{self.output_name}__attention"

        multihead_rules = [
            (R.get(self.output_name)(terms)[self.embed_dim, self.embed_dim] <= R.get(attention_name)(terms).T)
            | [Transformation.IDENTITY],
            R.get(self.output_name) / self.arity | [Transformation.IDENTITY],
        ]

        if self.num_heads > 1:
            qslice_name = f"{self.output_name}__qslice"
            kslice_name = f"{self.output_name}__kslice"
            vslice_name = f"{self.output_name}__vslice"
            attention = Attention(
                self.embed_dim // self.num_heads, attention_name, qslice_name, kslice_name, vslice_name, self.arity + 1
            )

            multihead_rules = [
                R.get(qslice_name) / (self.arity + 1) | [Transformation.IDENTITY],
                R.get(kslice_name) / (self.arity + 1) | [Transformation.IDENTITY],
                R.get(vslice_name) / (self.arity + 1) | [Transformation.IDENTITY],
                (R.get(self.output_name)(terms)[self.embed_dim, self.embed_dim] <= R.get(attention_name)("Y0", *terms))
                | [Transformation.IDENTITY, Aggregation.CONCAT],
                R.get(self.output_name) / self.arity | [Transformation.IDENTITY],
            ]

            for i in range(self.num_heads):
                size = self.embed_dim / self.num_heads
                metadata = [Transformation.SLICE(rows=(i * size, (i + 1) * size))]
                multihead_rules.append((R.get(qslice_name)(i, *terms) <= R.get(qproj_name)(terms)) | metadata)
                multihead_rules.append((R.get(kslice_name)(i, *terms) <= R.get(kproj_name)(terms)) | metadata)
                multihead_rules.append((R.get(vslice_name)(i, *terms) <= R.get(vproj_name)(terms)) | metadata)
        else:
            attention = Attention(
                self.embed_dim // self.num_heads, attention_name, qproj_name, kproj_name, vproj_name, self.arity
            )

        return [
            (R.get(qproj_name)(terms)[self.embed_dim, self.embed_dim] <= R.get(self.query_name)(terms))
            | [Transformation.IDENTITY],
            R.get(qproj_name) / self.arity | [Transformation.IDENTITY],
            (R.get(vproj_name)(terms)[self.embed_dim, self.vdim] <= R.get(self.value_name)(terms))
            | [Transformation.IDENTITY],
            R.get(vproj_name) / self.arity | [Transformation.IDENTITY],
            (R.get(kproj_name)(terms)[self.embed_dim, self.kdim] <= R.get(self.key_name)(terms))
            | [Transformation.IDENTITY],
            R.get(kproj_name) / self.arity | [Transformation.IDENTITY],
            *attention(),
            *multihead_rules,
        ]
