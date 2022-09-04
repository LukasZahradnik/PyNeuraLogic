from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Transformation, Aggregation, Combination
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class GATv2Conv(Module):
    r"""
    GATv2 layer from `"How Attentive are Graph Attention Networks?" <https://arxiv.org/abs/2105.14491>`_.

    Parameters
    ----------

    in_channels : int
        Input feature size.
    out_channels : int
        Output feature size.
    output_name : str
        Output (head) predicate name of the module.
    feature_name : str
        Feature predicate name to get features from.
    edge_name : str
        Edge predicate name to use for neighborhood relations.
    share_weights : bool
        Share weights in attention. Default: ``False``
    activation : Transformation
        Activation function of the output.
        Default: ``Transformation.IDENTITY``

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_name: str,
        feature_name: str,
        edge_name: str,
        share_weights: bool = False,
        activation: Transformation = Transformation.IDENTITY,
    ):
        self.output_name = output_name
        self.feature_name = feature_name
        self.edge_name = edge_name

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.share_weights = share_weights
        self.activation = activation

    def __call__(self):
        w1 = f"{self.output_name}__right"
        w2 = w1 if self.share_weights else f"{self.output_name}__left"

        attention = R.get(f"{self.output_name}__attention")
        attention_metadata = Metadata(transformation=Transformation.LEAKY_RELU)
        metadata = Metadata(
            transformation=Transformation.IDENTITY, aggregation=Aggregation.SUM, combination=Combination.PRODUCT
        )

        head = R.get(self.output_name)
        feature = R.get(self.feature_name)
        edge = R.get(self.edge_name)

        return [
            (
                attention(V.I, V.J)[self.out_channels, self.out_channels]
                <= (
                    feature(V.I)[w2 : self.out_channels, self.in_channels],
                    feature(V.J)[w1 : self.out_channels, self.in_channels],
                )
            )
            | attention_metadata,
            attention / 2 | Metadata(transformation=Transformation.SOFTMAX),
            (head(V.I) <= (attention(V.I, V.J), feature(V.J)[w1 : self.out_channels, self.in_channels], edge(V.J, V.I)))
            | metadata,
            head / 1 | Metadata(transformation=self.activation),
        ]
