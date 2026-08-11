from neuralogic.core.constructs.factories import R, V
from neuralogic.core.constructs.function import Aggregation, Combination, Transformation
from neuralogic.core.constructs.function.function import AggregationFunction, TransformationFunction
from neuralogic.core.constructs.metadata import Metadata
from neuralogic.nn.module.module import Module


class APPNPConv(Module):
    r"""
    Approximate Personalized Propagation of Neural Predictions layer from
    `"Predict then Propagate: Graph Neural Networks meet Personalized PageRank" <https://arxiv.org/abs/1810.05997>`_.
    Which can be expressed as:

    .. math::
        \mathbf{x}^{0}_i = \mathbf{x}_i

    .. math::
        \mathbf{x}^{k}_i = \alpha \cdot \mathbf{x}^0_i + (1 - \alpha) \cdot
        {agg}_{j \in \mathcal{N}(i)}(\mathbf{x}^{k - 1}_j)

    .. math::
        \mathbf{x}^{\prime}_i = act(\mathbf{x}^{K}_i)

    Where *act* is an activation function and *agg* aggregation function.


    The first part of the second equation that is ":math:`\alpha \cdot \mathbf{x}^0_i`" is expressed
    in the logic form as:

    .. code-block:: logtalk

        R.<output_name>__<k>(V.I) <= R.<feature_name>(V.I)[<alpha>].fixed()

    The second part of the second equation that is
    ":math:`(1 - \alpha) \cdot {agg}_{j \in \mathcal{N}(i)}(\mathbf{x}^{k - 1}_j)`" is expressed as:

    .. code-block:: logtalk

        R.<output_name>__<k>(V.I) <= (R.<output_name>__<k-1>(V.J)[1 - <alpha>].fixed(), R.<edge_name>(V.J, V.I))

    Examples
    --------

    The whole computation of this module
    (parametrized as :code:`APPNPConv("h1", "h0", "_edge", 3, 0.1, Transformation.SIGMOID)`) is as follows:

    .. code:: logtalk

        metadata = Metadata(transformation=Transformation.IDENTITY, aggregation=Aggregation.SUM)

        (R.h1__1(V.I) <= R.h0(V.I)[0.1].fixed()) | metadata
        (R.h1__1(V.I) <= (R.h0(V.J)[0.9].fixed(), R._edge(V.J, V.I))) | metadata
        R.h1__1/1 [Transformation.IDENTITY]

        (R.h1__2(V.I) <= <0.1> R.h0(V.I)) | metadata
        (R.h1__2(V.I) <= (<0.9> R.h1__1(V.J), R._edge(V.J, V.I))) | metadata
        R.h1__2/1 [Transformation.IDENTITY]

        (R.h1(V.I) <= <0.1> R.h0(V.I)) | metadata
        (R.h1(V.I) <= (<0.9> R.h1__2(V.J), R._edge(V.J, V.I))) | metadata
        R.h1 / 1 [Transformation.SIGMOID]


    Parameters
    ----------

    output_name : str
        Output (head) predicate name of the module.
    feature_name : str
        Feature predicate name to get features from.
    edge_name : str
        Edge predicate name to use for neighborhood relations.
    k : int
        Number of iterations
    alpha : float
        Teleport probability
    activation : TransformationFunction
        Activation function of the output.
        Default: ``Transformation.IDENTITY``
    aggregation : AggregationFunction
        Aggregation function of nodes' neighbors.
        Default: ``Aggregation.SUM``

    """

    def __init__(
        self,
        output_name: str,
        feature_name: str,
        edge_name: str,
        k: int,
        alpha: float,
        activation: TransformationFunction = Transformation.IDENTITY,
        aggregation: AggregationFunction = Aggregation.SUM,
        add_self_loops: bool | None = None,
        normalize: bool = True,
    ):
        self.output_name = output_name
        self.feature_name = feature_name
        self.edge_name = edge_name

        self.alpha = alpha
        self.k = k

        self.aggregation = aggregation
        self.activation = activation

        if add_self_loops is None:
            add_self_loops = normalize

        self.normalize = normalize
        self.add_self_loops = add_self_loops

    def __call__(self):
        head = R.get(self.output_name)(V.I)
        metadata = Metadata(aggregation=self.aggregation, combination=Combination.PRODUCT)
        edge = R.get(self.edge_name)
        edge_count = R.get(f"{self.output_name}__edge_count")
        feature = R.get(self.feature_name)

        self_loops = []
        normalization = []

        if self.add_self_loops:
            edge = R.get(f"{self.output_name}__edge")

            self_loops = [
                edge(V.I, V.I)[1.0].fixed(),
                edge(V.I, V.J) <= (R.get(self.edge_name)(V.I, V.J)),
            ]

        if self.normalize:
            # SUM, not COUNT: the degree is the sum of the incident edge values, which is what PyG's
            # `gcn_norm` takes and what makes an edge value the `edge_weight` it looks like. Identical at 1.0.
            count_metadata = Metadata(aggregation=Aggregation.SUM)

            normalization = [
                (edge_count(V.I, V.J) <= edge(V.X, V.J)) | count_metadata,
                (edge_count(V.I, V.J) <= edge(V.X, V.I)) | count_metadata,
                edge_count / 2 | Metadata(combination=Combination.PRODUCT, transformation=Transformation.INVERSE),
            ]

        def propagate(source):
            """One hop: the previous iterate carried along an edge, normalized, and scaled by 1 - alpha."""
            body = [source(V.J)[1 - self.alpha].fixed(), edge(V.J, V.I)]
            if self.normalize:
                body.append(Transformation.SQRT(edge_count(V.J, V.I)))
            return body

        rules = []
        for k in range(1, self.k):
            k_head = R.get(f"{self.output_name}__{k}")(V.I)
            previous = feature if k == 1 else R.get(f"{self.output_name}__{k - 1}")

            rules.append((k_head <= feature(V.I)[self.alpha].fixed()) | metadata)
            rules.append((k_head <= propagate(previous)) | metadata)

        last = feature if self.k == 1 else R.get(f"{self.output_name}__{self.k - 1}")

        return [
            *self_loops,
            *normalization,
            *rules,
            (head <= feature(V.I)[self.alpha].fixed()) | metadata,
            (head <= propagate(last)) | metadata,
            R.get(self.output_name) / 1 | Metadata(transformation=self.activation),
        ]
