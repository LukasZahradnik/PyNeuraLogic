from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Transformation, Aggregation, Combination
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class GCNConv(Module):
    r"""
    Graph Convolutional layer from
    `"Semi-supervised Classification with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_.

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
    activation : Transformation
        Activation function of the output.
        Default: ``Transformation.IDENTITY``
    aggregation : Aggregation
        Aggregation function of nodes' neighbors.
        Default: ``Aggregation.SUM``

    """

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
        metadata = Metadata(
            transformation=Transformation.IDENTITY, aggregation=self.aggregation, combination=Combination.PRODUCT
        )

        edge = R.get(f"{self.output_name}__edge")
        edge_count = R.get(f"{self.output_name}__edge_count")

        return [
            edge(V.I, V.I),
            (edge(V.I, V.J) <= (R.get(self.edge_name)(V.I, V.J))) | Metadata(transformation=Transformation.IDENTITY),
            edge / 2 | Metadata(transformation=Transformation.IDENTITY),
            (edge_count(V.I, V.J) <= edge(V.J, V.X))
            | Metadata(transformation=Transformation.IDENTITY, aggregation=Aggregation.COUNT),
            (edge_count(V.I, V.J) <= edge(V.I, V.X))
            | Metadata(transformation=Transformation.IDENTITY, aggregation=Aggregation.COUNT),
            edge_count / 2 | Metadata(combination=Combination.PRODUCT, transformation=Transformation.INVERSE),
            (head <= (R.get(self.feature_name)(V.J), edge(V.J, V.I), Transformation.SQRT(edge_count(V.J, V.I))))
            | metadata,
            R.get(self.output_name) / 1 | Metadata(transformation=self.activation),
        ]
