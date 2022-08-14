from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Transformation, Aggregation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class GINConv(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_name: str,
        feature_name: str,
        edge_name: str,
        activation: Transformation = Transformation.IDENTITY,
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
        embed = R.get(f"embed__{self.output_name}")

        metadata = Metadata(transformation=Transformation.IDENTITY, aggregation=self.aggregation)

        return [
            (head <= (R.get(self.feature_name)(V.J), R.get(self.edge_name)(V.J, V.I))) | metadata,
            (embed(V.I) <= R.get(self.feature_name)(V.I)) | metadata,
            (head <= embed(V.I)[self.in_channels, self.in_channels]) | Metadata(transformation=self.activation),
            embed / 1 | Metadata(transformation=Transformation.IDENTITY),
            R.get(self.output_name) / 1 | Metadata(transformation=self.activation),
        ]
