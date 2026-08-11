from neuralogic.core.constructs.factories import R, V
from neuralogic.core.constructs.function import Aggregation, Combination, Transformation
from neuralogic.core.constructs.function.function import AggregationFunction, TransformationFunction
from neuralogic.core.constructs.metadata import Metadata
from neuralogic.nn.module.module import Module


class GCNConv(Module):
    r"""
    Graph Convolutional layer from
    `"Semi-supervised Classification with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_.

    .. math::
        \mathbf{X}^{\prime} = act \left( \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}
        \mathbf{X} \mathbf{W} \right)

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}`. The body combines by product, so an edge's
    value is PyG's ``edge_weight``: it scales the message it carries and it enters the degree. Note
    ``__edge_count`` is a weighted **in-**degree despite its name - the two rules sum over ``V.X`` in the
    *source* position, and they sum the edge values rather than counting the edges, both of which is what
    ``gcn_norm`` does. Neither is visible on an undirected graph given both directions at ``1.0``, where
    in-degree is out-degree and summing ones is counting them, which is why the name survived being either.

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
    activation : TransformationFunction
        Activation function of the output.
        Default: ``Transformation.IDENTITY``
    aggregation : AggregationFunction
        Aggregation function of nodes' neighbors.
        Default: ``Aggregation.SUM``
    add_self_loops : bool | None
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
        activation: TransformationFunction = Transformation.IDENTITY,
        aggregation: AggregationFunction = Aggregation.SUM,
        add_self_loops: bool | None = None,
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
        metadata = Metadata(aggregation=self.aggregation, combination=Combination.PRODUCT)

        edge = R.get(self.edge_name)
        edge_count = R.get(f"{self.output_name}__edge_count")

        self_loops = []
        normalization = []
        body = [R.get(self.feature_name)(V.J), edge(V.J, V.I)]

        if self.add_self_loops:
            edge = R.get(f"{self.output_name}__edge")

            self_loops = [
                edge(V.I, V.I)[1.0].fixed(),
                edge(V.I, V.J) <= (R.get(self.edge_name)(V.I, V.J)),
            ]

        if self.normalize:
            # SUM rather than COUNT, so the degree is the *sum* of the incident edge values as PyG's
            # `gcn_norm` takes it. At the natural 1.0 the two are the same number - summing ones is counting
            # them - so this only shows up once a graph carries real edge weights, and then it is the
            # difference between accepting them and ignoring them.
            count_metadata = Metadata(aggregation=Aggregation.SUM)
            body = [R.get(self.feature_name)(V.J), edge(V.J, V.I), Transformation.SQRT(edge_count(V.J, V.I))]

            normalization = [
                (edge_count(V.I, V.J) <= edge(V.X, V.J)) | count_metadata,
                (edge_count(V.I, V.J) <= edge(V.X, V.I)) | count_metadata,
                edge_count / 2 | Metadata(combination=Combination.PRODUCT, transformation=Transformation.INVERSE),
            ]

        return [
            *self_loops,
            *normalization,
            (head <= body) | metadata,
            R.get(self.output_name) / 1 | Metadata(transformation=self.activation),
        ]
