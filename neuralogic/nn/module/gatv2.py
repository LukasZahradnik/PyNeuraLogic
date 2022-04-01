from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.enums import Activation, Aggregation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class GATv2Conv(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_name: str,
        feature_name: str,
        edge_name: str,
        share_weights: bool = False,
        activation: Activation = Activation.IDENTITY,
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

        attention_metadata = Metadata(activation=Activation.LEAKY_RELU)
        metadata = Metadata(activation="product-identity", aggregation=Aggregation.SUM)
        attention = R.get(f"{self.output_name}__attention")

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
            attention / 2 | Metadata(activation=Activation.SOFTMAX),
            (head(V.I) <= (attention(V.I, V.J), feature(V.J)[w1 : self.out_channels, self.in_channels], edge(V.J, V.I)))
            | metadata,
            head / 1 | Metadata(activation=self.activation),
        ]
