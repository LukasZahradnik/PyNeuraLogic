import math
from typing import Optional

from neuralogic.core.constructs.function import Transformation, Combination, Aggregation
from neuralogic.core.constructs.factories import R
from neuralogic.nn.module.module import Module


class Attention(Module):
    r"""
    A single-head attention module based on `"Attention Is All You Need" <https://arxiv.org/abs/1706.03762>`_.

    Parameters
    ----------

    embed_dim : int
        The number of expected features.
    output_name : str
        Output (head) predicate name of the module.
    query_name : str
        The name of the queries predicate.
    key_name : str
        The name of the keys predicate.
    value_name : str
        The name of the values predicate.
    mask_name : str, optional
        The name of the input mask predicate. Default: ``None``
    arity : int
        Arity of the input and output predicates. Default: ``1``
    """

    def __init__(
        self,
        embed_dim: int,
        output_name: str,
        query_name: str,
        key_name: str,
        value_name: str,
        mask_name: Optional[str] = None,
        arity: int = 1,
    ):
        self.embed_dim = embed_dim
        self.output_name = output_name
        self.query_name = query_name
        self.key_name = key_name
        self.value_name = value_name
        self.mask_name = mask_name
        self.arity = arity

    def __call__(self):
        terms = [f"X{i}" for i in range(self.arity - 1)]

        k_terms = [*terms, "Y"]
        h_terms = [*terms, "X", "Y"]
        q_terms = [*terms, "X"]

        d_k = 1 / math.sqrt(self.embed_dim)

        dk_rel = R.get(f"{self.output_name}__dk")
        dot_rel = R.get(f"{self.output_name}__dot")

        metadata = [Combination.PRODUCT, Transformation.IDENTITY, Aggregation.SOFTMAX(agg_terms=["Y"])]
        out_metadata = [Combination.PRODUCT, Aggregation.SUM, Transformation.IDENTITY]

        attention_product_rules = [
            (dot_rel(h_terms) <= (dk_rel, R.get(self.key_name)(k_terms).T, R.get(self.query_name)(q_terms))) | metadata,
            dot_rel / (self.arity + 1) | [Transformation.IDENTITY],
        ]

        if self.mask_name is not None:
            attention_product_rules[0].body.append(R.hidden.get(self.mask_name)(h_terms))

        return [
            dk_rel[d_k].fixed(),
            *attention_product_rules,
            (R.get(self.output_name)(q_terms) <= (dot_rel(h_terms), R.get(self.value_name)(k_terms))) | out_metadata,
            R.get(self.output_name) / self.arity | [Transformation.IDENTITY],
        ]


class MultiheadAttention(Module):
    r"""
    A multi-head attention module based on `"Attention Is All You Need" <https://arxiv.org/abs/1706.03762>`_.

    Parameters
    ----------

    embed_dim : int
        The number of expected features.
    num_heads : int
        The number of heads.
    output_name : str
        Output (head) predicate name of the module.
    query_name : str
        The name of the queries predicate.
    key_name : str
        The name of the keys predicate.
    value_name : str
        The name of the values predicate.
    vdim : int
        Total number of features for values.
    kdim : int
        Total number of features for keys.
    mask_name : str, optional
        The name of the input mask predicate. Default: ``None``
    arity : int
        Arity of the input and output predicates. Default: ``1``
    """

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
        mask_name: Optional[str] = None,
        arity: int = 1,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_name = output_name
        self.queries = query_name
        self.keys = key_name
        self.values = value_name
        self.vdim = vdim if vdim is not None else embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.mask_name = mask_name
        self.arity = arity

    def __call__(self):
        terms = [f"X{i}" for i in range(self.arity)]
        dim = self.embed_dim

        q_weight = f"{self.output_name}_qw"
        k_weight = f"{self.output_name}_kw"
        v_weight = f"{self.output_name}_vw"

        q_proj_name = f"{self.output_name}__qproj"
        k_proj_name = f"{self.output_name}__kproj"
        v_proj_name = f"{self.output_name}__vproj"

        q_proj = R.get(q_proj_name)
        k_proj = R.get(k_proj_name)
        v_proj = R.get(v_proj_name)
        output_rel = R.get(self.output_name)

        attention_name = f"{self.output_name}__attention"

        attention = Attention(
            dim // self.num_heads, attention_name, q_proj_name, k_proj_name, v_proj_name, self.mask_name, self.arity
        )

        if self.num_heads != 1:
            size = self.embed_dim / self.num_heads
            attention.arity += 1

            attention_concat = []
            multihead_rules = [
                q_proj / (self.arity + 1) | [Transformation.IDENTITY],
                k_proj / (self.arity + 1) | [Transformation.IDENTITY],
                v_proj / (self.arity + 1) | [Transformation.IDENTITY],
                output_rel / self.arity | [Transformation.IDENTITY],
            ]

            for i in range(self.num_heads):
                meta = [Transformation.SLICE(rows=(i * size, (i + 1) * size))]
                multihead_rules.append((q_proj(i, *terms) <= R.get(self.queries)(terms)[q_weight:dim, dim]) | meta)
                multihead_rules.append((v_proj(i, *terms) <= R.get(self.values)(terms)[v_weight:dim, self.vdim]) | meta)
                multihead_rules.append((k_proj(i, *terms) <= R.get(self.keys)(terms)[k_weight:dim, self.kdim]) | meta)
                attention_concat.append(R.get(attention_name)(i, *terms))

            multihead_rules.append(
                (output_rel(terms)[dim, dim] <= attention_concat) | [Transformation.IDENTITY, Combination.CONCAT]
            )
        else:
            multihead_rules = [
                (q_proj(terms)[q_weight:dim, dim] <= R.get(self.queries)(terms)) | [Transformation.IDENTITY],
                q_proj / self.arity | [Transformation.IDENTITY],
                (v_proj(terms)[v_weight:dim, self.vdim] <= R.get(self.values)(terms)) | [Transformation.IDENTITY],
                v_proj / self.arity | [Transformation.IDENTITY],
                (k_proj(terms)[k_weight:dim, self.kdim] <= R.get(self.keys)(terms)) | [Transformation.IDENTITY],
                k_proj / self.arity | [Transformation.IDENTITY],
                (output_rel(terms)[dim, dim] <= R.get(attention_name)(terms)) | [Transformation.IDENTITY],
                output_rel / self.arity | [Transformation.IDENTITY],
            ]

        return [*attention(), *multihead_rules]
