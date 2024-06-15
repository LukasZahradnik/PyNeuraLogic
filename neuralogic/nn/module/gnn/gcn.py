from typing import Optional

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
    add_self_loops : Optional[bool]
        Add self loops if either set to `True` or `None` (if `normalize` is `True`).
        Default: ``None``
    normalize : bool
        Add normalization.
        Default : ``True``
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
        add_self_loops: Optional[bool] = None,
        normalize: bool = True,
    ):
        self.output_name = output_name
        self.feature_name = feature_name
        self.edge_name = edge_name

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.aggregation = aggregation
        self.activation = activation

        if add_self_loops is None:
            add_self_loops = normalize

        self.normalize = normalize
        self.add_self_loops = add_self_loops

    def __call__(self):
        head = R.get(self.output_name)(V.I)[self.out_channels, self.in_channels]
        metadata = Metadata(
            transformation=Transformation.IDENTITY, aggregation=self.aggregation, combination=Combination.PRODUCT
        )

        id_metadata = Metadata(transformation=Transformation.IDENTITY)

        edge = R.get(self.edge_name)
        edge_count = R.get(f"{self.output_name}__edge_count")

        self_loops = []
        normalization = []
        body = [R.get(self.feature_name)(V.J), edge(V.J, V.I)]

        if self.add_self_loops:
            edge = R.get(f"{self.output_name}__edge")

            self_loops = [
                edge(V.I, V.I)[1.0].fixed(),
                (edge(V.I, V.J) <= (R.get(self.edge_name)(V.I, V.J))) | id_metadata,
                edge / 2 | id_metadata,
            ]

        if self.normalize:
            count_metadata = Metadata(transformation=Transformation.IDENTITY, aggregation=Aggregation.COUNT)
            body = [R.get(self.feature_name)(V.J), edge(V.J, V.I), Transformation.SQRT(edge_count(V.J, V.I))]

            normalization = [
                (edge_count(V.I, V.J) <= edge(V.J, V.X)) | count_metadata,
                (edge_count(V.I, V.J) <= edge(V.I, V.X)) | count_metadata,
                edge_count / 2 | Metadata(combination=Combination.PRODUCT, transformation=Transformation.INVERSE),
            ]

        return [
            *self_loops,
            *normalization,
            (head <= body) | metadata,
            R.get(self.output_name) / 1 | Metadata(transformation=self.activation),
        ]
