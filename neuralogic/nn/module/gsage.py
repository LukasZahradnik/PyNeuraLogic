from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.enums import Activation, Aggregation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class SAGEConv(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_name: str,
        feature_name: str,
        edge_name: str,
        activation: Activation = Activation.IDENTITY,
        aggregation: Aggregation = Aggregation.SUM,
    ):
        self.output_name = output_name
        self.feature_name = feature_name
        self.edge_name = edge_name

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.aggregation = aggregation
        self.activation = activation

    def __call__(self):
        head = R.get(self.output_name)(V.I)[self.out_channels, self.in_channels]
        metadata = Metadata(activation=Activation.IDENTITY, aggregation=self.aggregation)

        return [
            (head <= (R.get(self.feature_name)(V.J), R.get(self.edge_name)(V.J, V.I))) | metadata,
            (head <= R.get(self.feature_name)(V.I)) | metadata,
            R.get(self.output_name) / 1 | Metadata(activation=self.activation),
        ]
