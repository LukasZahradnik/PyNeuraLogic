from typing import Any

from neuralogic.core.constructs.factories import R, V
from neuralogic.core.constructs.function import Aggregation, Transformation
from neuralogic.core.constructs.function.function import AggregationFunction, TransformationFunction
from neuralogic.core.constructs.metadata import Metadata
from neuralogic.nn.module.module import Module


class GINConv(Module):
    """
    Implements the Graph Isomorphism Network (GIN) convolution layer.
    GIN is a powerful GNN layer that can distinguish between different graph structures.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_name: str,
        feature_name: str,
        edge_name: str,
        activation: TransformationFunction = Transformation.IDENTITY,
        aggregation: AggregationFunction = Aggregation.SUM,
    ):
        """
        Parameters
        ----------
        in_channels : int
            Number of input features.
        out_channels : int
            Number of output features.
        output_name : str
            Name of the output relation.
        feature_name : str
            Name of the input feature relation.
        edge_name : str
            Name of the edge relation.
        activation : TransformationFunction, optional
            Activation function to use. Default: Transformation.IDENTITY.
        aggregation : AggregationFunction, optional
            Aggregation function to use. Default: Aggregation.SUM.
        """
        self.output_name = output_name
        self.feature_name = feature_name
        self.edge_name = edge_name

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.aggregation = aggregation
        self.activation = activation

    def __call__(self) -> list[Any]:
        """
        Generates the rules for GIN convolution.

        Returns
        -------
        list
            A list of rules defining the GIN convolution.
        """
        head = R.get(self.output_name)(V.I)[self.out_channels, self.in_channels]
        embed = R.get(f"embed__{self.output_name}")

        metadata = Metadata(aggregation=self.aggregation)

        return [
            (head <= (R.get(self.feature_name)(V.J), R.get(self.edge_name)(V.J, V.I))) | metadata,
            (embed(V.I) <= R.get(self.feature_name)(V.I)) | metadata,
            (head <= embed(V.I)[self.in_channels, self.in_channels]) | Metadata(transformation=self.activation),
            R.get(self.output_name) / 1 | Metadata(transformation=self.activation),
        ]
