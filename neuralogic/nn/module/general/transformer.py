from typing import Optional

from neuralogic.core.constructs.function import Transformation
from neuralogic.core.constructs.factories import R
from neuralogic.nn.module.module import Module
from neuralogic.nn.module.general.mlp import MLP
from neuralogic.nn.module.general.attention import MultiheadAttention


class Transformer(Module):
    r"""
    A transformer module based on `"Attention Is All You Need" <https://arxiv.org/abs/1706.03762>`_.

    Parameters
    ----------

    input_dim : int
        The number of expected features.
    num_heads : int
        The number of heads in the multi-head attention module.
    dim_feedforward : int
        The dimension of the feedforward network.
    output_name : str
        Output (head) predicate name of the module.
    src_name : str
        The name of the predicate of the input to the encoder.
    tgt_name : str
        The name of the predicate of the input to the decoder.
    src_mask_name : str, optional
        The name of the predicate of the encoder input mask. Default: ``None``
    tgt_mask_name : str, optional
        The name of the predicate of the decoder input mask. Default: ``None``
    memory_mask_name : str, optional
        The name of the predicate of the encoder output mask. Default: ``None``
    arity : int
        Arity of the input and output predicate. Default: ``1``
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dim_feedforward: int,
        output_name: str,
        src_name: str,
        tgt_name: str,
        src_mask_name: Optional[str] = None,
        tgt_mask_name: Optional[str] = None,
        memory_mask_name: Optional[str] = None,
        arity: int = 1,
    ):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.output_name = output_name
        self.src_name = src_name
        self.tgt_name = tgt_name
        self.src_mask_name = src_mask_name
        self.tgt_mask_name = tgt_mask_name
        self.memory_mask_name = memory_mask_name
        self.arity = arity

    def __call__(self):
        encoder = TransformerEncoder(
            self.input_dim,
            self.num_heads,
            self.dim_feedforward,
            f"{self.output_name}__encoder",
            self.src_name,
            self.src_mask_name,
            self.arity,
        )

        decoder = TransformerDecoder(
            self.input_dim,
            self.num_heads,
            self.dim_feedforward,
            self.output_name,
            self.tgt_name,
            f"{self.output_name}__encoder",
            self.tgt_mask_name,
            self.memory_mask_name,
            self.arity,
        )

        return [
            *encoder(),
            *decoder(),
        ]


class EncoderBlock(Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dim_feedforward: int,
        output_name: str,
        query_name: str,
        key_name: str,
        value_name: str,
        mask_name: Optional[str] = None,
        arity: int = 1,
        mlp: bool = True,
    ):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.output_name = output_name
        self.query_name = query_name
        self.key_name = key_name
        self.value_name = value_name
        self.mask_name = mask_name
        self.arity = arity
        self.mlp = mlp

    def __call__(self):
        terms = [f"X{i}" for i in range(self.arity)]

        attn_name = f"{self.output_name}__mhattn"
        norm_name = f"{self.output_name}__norm"
        mlp_name = f"{self.output_name}__mlp"

        output_rel = R.get(self.output_name)
        dim = self.input_dim
        data_name = self.query_name

        attention = MultiheadAttention(
            dim,
            self.num_heads,
            attn_name,
            self.query_name,
            self.key_name,
            self.value_name,
            mask_name=self.mask_name,
            arity=self.arity,
        )

        if self.mlp:
            dims = [dim, self.dim_feedforward, dim]
            mlp = MLP(dims, mlp_name, norm_name, [Transformation.RELU, Transformation.IDENTITY], self.arity)

            return [
                *attention(),
                (R.get(norm_name)(terms) <= (R.get(attn_name)(terms), R.get(data_name)(terms))) | [Transformation.NORM],
                R.get(norm_name) / self.arity | [Transformation.IDENTITY],
                *mlp(),
                (output_rel(terms) <= (R.get(norm_name)(terms), R.get(mlp_name)(terms))) | [Transformation.NORM],
                output_rel / self.arity | [Transformation.IDENTITY],
            ]

        return [
            *attention(),
            (output_rel(terms) <= (R.get(attn_name)(terms), R.get(data_name)(terms))) | [Transformation.NORM],
            output_rel / self.arity | [Transformation.IDENTITY],
        ]


class TransformerEncoder(EncoderBlock):
    r"""
    A transformer encoder module based on `"Attention Is All You Need" <https://arxiv.org/abs/1706.03762>`_.

    Parameters
    ----------

    input_dim : int
        The number of expected features.
    num_heads : int
        The number of heads in the multi-head attention module.
    dim_feedforward : int
        The dimension of the feedforward network.
    output_name : str
        Output (head) predicate name of the module.
    input_name : str
        The name of the predicate of the input sequence.
    mask_name : str, optional
        The name of the predicate of the input sequence mask. Default: ``None``
    arity : int
        Arity of the input and output predicate. Default: ``1``
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dim_feedforward: int,
        output_name: str,
        input_name: str,
        mask_name: Optional[str] = None,
        arity: int = 1,
    ):
        super().__init__(
            input_dim,
            num_heads,
            dim_feedforward,
            output_name,
            input_name,
            input_name,
            input_name,
            mask_name,
            arity,
            True,
        )


class TransformerDecoder(Module):
    r"""
    A transformer decoder module based on `"Attention Is All You Need" <https://arxiv.org/abs/1706.03762>`_.

    Parameters
    ----------

    input_dim : int
        The number of expected features.
    num_heads : int
        The number of heads in the multi-head attention module.
    dim_feedforward : int
        The dimension of the feedforward network.
    output_name : str
        Output (head) predicate name of the module.
    input_name : str
        The name of the predicate of the input sequence.
    input_name : str
        The name of the input encoder.
    mask_name : str, optional
        The name of the predicate of the decoder input sequence mask. Default: ``None``
    memory_mask_name : str, optional
        The name of the predicate of the encoder output mask. Default: ``None``
    arity : int
        Arity of the input and output predicate. Default: ``1``
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dim_feedforward: int,
        output_name: str,
        input_name: str,
        encoder_name: str,
        mask_name: Optional[str] = None,
        memory_mask_name: Optional[str] = None,
        arity: int = 1,
    ):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.output_name = output_name
        self.input_name = input_name
        self.encoder_name = encoder_name
        self.mask_name = mask_name
        self.memory_mask_name = memory_mask_name
        self.arity = arity

    def __call__(self):
        data_name = self.input_name
        dim = self.input_dim

        tmp_encoder_out = f"{self.output_name}__decoder"
        encoder_name = self.encoder_name
        mlp_dim = self.dim_feedforward

        enc_block_one = EncoderBlock(
            dim,
            self.num_heads,
            mlp_dim,
            tmp_encoder_out,
            data_name,
            data_name,
            data_name,
            self.mask_name,
            self.arity,
            False,
        )

        enc_block_two = EncoderBlock(
            dim,
            self.num_heads,
            mlp_dim,
            self.output_name,
            tmp_encoder_out,
            encoder_name,
            encoder_name,
            self.memory_mask_name,
            self.arity,
        )

        return [
            *enc_block_one(),
            *enc_block_two(),
        ]
