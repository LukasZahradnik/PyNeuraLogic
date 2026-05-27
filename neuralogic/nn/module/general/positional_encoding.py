import math
from typing import Any

import numpy as np

from neuralogic.core.constructs.factories import R
from neuralogic.core.constructs.function import Combination
from neuralogic.nn.module.module import Module


class PositionalEncoding(Module):
    """
    Implements positional encoding for logic sequences.
    It generates a set of rules that add fixed or learnable positional embeddings to the input features.
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int,
        output_name: str,
        input_name: str,
        arity: int = 1,
        learnable: bool = False,
    ):
        """
        Parameters
        ----------
        embed_dim : int
            The dimension of the positional embedding.
        max_len : int
            The maximum length of the sequence.
        output_name : str
            The name of the output relation (with positional encoding added).
        input_name : str
            The name of the input relation to add encoding to.
        arity : int, optional
            The arity of the input/output relations. Default: 1.
        learnable : bool, optional
            Whether the positional encodings should be learnable parameters. Default: False.
        """
        self.embed_dim = embed_dim
        self.max_len = max_len

        self.output_name = output_name
        self.input_name = input_name

        self.arity = arity
        self.learnable = learnable

    def __call__(self) -> list[Any]:
        """
        Generates the rules for positional encoding.

        Returns
        -------
        list
            A list of rules defining the positional encoding nodes and the final combined output.
        """
        terms = [f"X{i}" for i in range(self.arity - 1)]
        all_terms = [f"X{i}" for i in range(self.arity)]

        position = np.arange(0, self.max_len).reshape((self.max_len, 1))
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * (-math.log(1000.0) / self.embed_dim))

        pe = np.zeros((self.max_len, self.embed_dim))
        mul = position * div_term

        pe[:, 0::2] = np.sin(mul)
        pe[:, 1::2] = np.cos(mul)

        pe_rel = R.get(f"{self.output_name}__pe")
        out_rel = R.get(self.output_name)
        in_rel = R.get(self.input_name)

        if self.learnable:
            rules = [pe_rel(*terms, i)[row] for i, row in enumerate(pe)]
        else:
            rules = [pe_rel(*terms, i)[row].fixed() for i, row in enumerate(pe)]

        rules.append((out_rel(all_terms) <= (pe_rel(all_terms), in_rel(all_terms))) | [Combination.SUM])

        return rules
