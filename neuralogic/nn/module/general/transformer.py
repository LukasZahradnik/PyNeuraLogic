from neuralogic.core.constructs.function import Transformation
from neuralogic.core.constructs.factories import R
from neuralogic.nn.module.module import Module
from neuralogic.nn.module.general.mlp import MLP
from neuralogic.nn.module.general.attention import MultiheadAttention


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
        self.arity = arity
        self.mlp = mlp

    def __call__(self):
        terms = [f"X{i}" for i in range(self.arity)]

        attn_name = f"{self.output_name}__mhattn"
        norm_name = f"{self.output_name}__norm"

        output_name = self.output_name
        dim = self.input_dim
        data_name = self.query_name

        attention = MultiheadAttention(
            dim, self.num_heads, attn_name, self.query_name, self.key_name, self.value_name, arity=self.arity
        )

        if self.mlp:
            dims = [dim, self.dim_feedforward, self.dim_feedforward, dim]
            mlp = MLP(dims, output_name, norm_name, activation=[Transformation.RELU, Transformation.NORM])

            return [
                *mlp(),
                *attention(),
                (R.get(norm_name)(terms) <= (R.get(attn_name)(terms), R.get(data_name)(terms))) | [Transformation.NORM],
                R.get(norm_name) / self.arity | [Transformation.IDENTITY],
            ]

        return [
            *attention(),
            (R.get(output_name)(terms) <= (R.get(attn_name)(terms), R.get(data_name)(terms))) | [Transformation.NORM],
            R.get(output_name) / self.arity | [Transformation.IDENTITY],
        ]


class TransformerEncoder(EncoderBlock):
    def __init__(
        self, input_dim: int, num_heads: int, dim_feedforward: int, output_name: str, input_name: str, arity: int = 1
    ):
        super().__init__(
            input_dim, num_heads, dim_feedforward, output_name, input_name, input_name, input_name, arity, True
        )


class TransformerDecoder(Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dim_feedforward: int,
        output_name: str,
        input_name: str,
        encoder_name: str,
        arity: int = 1,
    ):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.output_name = output_name
        self.input_name = input_name
        self.encoder_name = encoder_name
        self.arity = arity

    def __call__(self):
        data_name = self.input_name
        dim = self.input_dim

        tmp_encoder_out = f"{self.output_name}__encoder"
        encoder_name = self.encoder_name
        mlp_dim = self.dim_feedforward

        enc_block_one = EncoderBlock(
            dim, self.num_heads, mlp_dim, tmp_encoder_out, data_name, data_name, data_name, self.arity, False
        )

        enc_block_two = EncoderBlock(
            dim, self.num_heads, mlp_dim, self.output_name, tmp_encoder_out, encoder_name, encoder_name, self.arity
        )

        return [
            *enc_block_one(),
            *enc_block_two(),
        ]
