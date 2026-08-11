from neuralogic.core.constructs.factories import R, V
from neuralogic.core.constructs.function import Aggregation, Combination, Transformation
from neuralogic.core.constructs.function.function import AggregationFunction, TransformationFunction
from neuralogic.core.constructs.metadata import Metadata
from neuralogic.nn.module.module import Module


class SGConv(Module):
    r"""
    Simple Graph Convolutional layer from `"Simplifying Graph Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_.
    Which can be expressed as:

    .. math::
        \mathbf{X}^{\prime} = act \left( {\left( \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^k \mathbf{X} \mathbf{W} \right)

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}`, *act* is an activation function and *W* is a
    learnable parameter. Each of the *k* hops walks one edge and carries one normalization factor, so the
    equation is translated into the logic form as:

    .. code:: logtalk

        R.<output_name>__edge(V.I, V.I)[1.0].fixed()
        R.<output_name>__edge(V.I, V.J) <= R.<edge_name>(V.I, V.J)

        (R.<output_name>__edge_count(V.I, V.J) <= R.<output_name>__edge(V.J, V.X)) | [Aggregation.COUNT]
        (R.<output_name>__edge_count(V.I, V.J) <= R.<output_name>__edge(V.I, V.X)) | [Aggregation.COUNT]
        R.<output_name>__edge_count / 2 | [Combination.PRODUCT, Transformation.INVERSE]

        (R.<output_name>(V.I<0>)[<W>] <= (
            R.<feature_name>(V.I<k>),
            R.<output_name>__edge(V.I<1>, V.I<0>), Transformation.SQRT(R.<output_name>__edge_count(V.I<1>, V.I<0>)),
            ...,
            R.<output_name>__edge(V.I<k>, V.I<k-1>), Transformation.SQRT(R.<output_name>__edge_count(V.I<k>, V.I<k-1>)),
        )) | [<aggregation>, Combination.PRODUCT]

        R.<output_name> / 1 | [<activation>]

    The body combines by product, which is what makes the normalization factors scale the features rather
    than shift them - and it is also why the edge can stay an ordinary valued atom rather than a hidden one:
    ``1.0`` is the identity of a product, so the natural spelling of a graph costs nothing.

    An edge given some other value does scale the message it carries, but that is **not** the same as PyG's
    ``edge_weight``: ``gcn_norm`` takes the degree as the *sum* of the incident edge weights, while the
    counting rules above count groundings and ignore their values. The two therefore agree exactly at
    ``1.0`` and nowhere else - with ``normalize=False``, where no degree is involved, an edge value scales
    the sum linearly as one would expect.

    Examples
    --------

    Module parametrized as :code:`SGConv(2, 3, "h1", "h0", "_edge", 1)` translates into:

    .. code:: logtalk

        R.h1__edge(V.I, V.I)[1.0].fixed()
        R.h1__edge(V.I, V.J) <= R._edge(V.I, V.J)
        (R.h1__edge_count(V.I, V.J) <= R.h1__edge(V.J, V.X)) | [Aggregation.COUNT]
        (R.h1__edge_count(V.I, V.J) <= R.h1__edge(V.I, V.X)) | [Aggregation.COUNT]
        R.h1__edge_count / 2 | [Combination.PRODUCT, Transformation.INVERSE]
        (R.h1(V.I0)[3, 2] <= (R.h0(V.I1), R.h1__edge(V.I1, V.I0), Transformation.SQRT(R.h1__edge_count(V.I1, V.I0)))) | [Aggregation.SUM, Combination.PRODUCT]
        R.h1 / 1 | [Transformation.IDENTITY]

    Setting :code:`normalize=False` and :code:`add_self_loops=False` drops both groups of extra rules and
    leaves the plain walk-and-sum this module used to be - which is not what PyG computes.


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
    k : int
        Number of hops.
        Default: ``1``
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
        Add symmetric normalization.
        Default: ``True``

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_name: str,
        feature_name: str,
        edge_name: str,
        k: int = 1,
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

        self.k = k

        self.aggregation = aggregation
        self.activation = activation

        if add_self_loops is None:
            add_self_loops = normalize

        self.normalize = normalize
        self.add_self_loops = add_self_loops

    def __call__(self):
        head = R.get(self.output_name)(V.I0)[self.out_channels, self.in_channels]
        metadata = Metadata(
            aggregation=self.aggregation, combination=Combination.PRODUCT, duplicate_grounding=True
        )

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
            count_metadata = Metadata(aggregation=Aggregation.COUNT)

            normalization = [
                (edge_count(V.I, V.J) <= edge(V.J, V.X)) | count_metadata,
                (edge_count(V.I, V.J) <= edge(V.I, V.X)) | count_metadata,
                edge_count / 2 | Metadata(combination=Combination.PRODUCT, transformation=Transformation.INVERSE),
            ]

        body = [feature(f"I{self.k}")]
        for near, far in zip(range(self.k), range(1, self.k + 1)):
            body.append(edge(f"I{far}", f"I{near}"))
            if self.normalize:
                body.append(Transformation.SQRT(edge_count(f"I{far}", f"I{near}")))

        return [
            *self_loops,
            *normalization,
            (head <= body) | metadata,
            R.get(self.output_name) / 1 | Metadata(transformation=self.activation),
        ]
