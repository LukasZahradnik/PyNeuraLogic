from neuralogic.core.constructs.factories import R, V
from neuralogic.core.constructs.function import Aggregation, Combination, Transformation
from neuralogic.core.constructs.function.function import AggregationFunction, TransformationFunction
from neuralogic.core.constructs.metadata import Metadata
from neuralogic.nn.module.module import Module


class TAGConv(Module):
    r"""
    Topology Adaptive Graph Convolutional layer from
    `"Topology Adaptive Graph Convolutional Networks" <https://arxiv.org/abs/1710.10370>`_.
    Which can be expressed as:

    .. math::
        \mathbf{X}^{\prime} = act \left( \sum_{k=0}^K {\left( \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \right)}^k \mathbf{X} \mathbf{W}_k \right)

    Where *act* is an activation function and *Wk* are learnable parameters, one per hop. Note this
    normalization is **not** the one :class:`~neuralogic.nn.module.gnn.gcn.GCNConv` and
    :class:`~neuralogic.nn.module.gnn.sg.SGConv` use: there are no self-loops, so the degree is the plain
    degree rather than one more than it, matching PyG's ``add_self_loops=False``. The equation is translated
    into the logic form as:

    .. code:: logtalk

        (R.<output_name>__edge_count(V.I, V.J) <= R.<edge_name>(V.J, V.X)) | [Aggregation.COUNT]
        (R.<output_name>__edge_count(V.I, V.J) <= R.<edge_name>(V.I, V.X)) | [Aggregation.COUNT]
        R.<output_name>__edge_count / 2 | [Combination.PRODUCT, Transformation.INVERSE]

        (R.<output_name>(V.I0)[<W0>] <= R.<feature_name>(V.I0)) | [<aggregation>, Combination.PRODUCT]
        (R.<output_name>(V.I0)[<W1>] <= (R.<feature_name>(V.I1), R.<edge_name>(V.I1, V.I0), Transformation.SQRT(R.<output_name>__edge_count(V.I1, V.I0)))) | [<aggregation>, Combination.PRODUCT]
        ...
        R.<output_name> / 1 | [<activation>]

    Examples
    --------

    Module parametrized as :code:`TAGConv(1, 2, "h1", "h0", "_edge", 1)` translates into:

    .. code:: logtalk

        (R.h1__edge_count(V.I, V.J) <= R._edge(V.J, V.X)) | [Aggregation.COUNT]
        (R.h1__edge_count(V.I, V.J) <= R._edge(V.I, V.X)) | [Aggregation.COUNT]
        R.h1__edge_count / 2 | [Combination.PRODUCT, Transformation.INVERSE]
        (R.h1(V.I0)[2, 1] <= R.h0(V.I0)) | [Aggregation.SUM, Combination.PRODUCT]
        (R.h1(V.I0)[2, 1] <= (R.h0(V.I1), R._edge(V.I1, V.I0), Transformation.SQRT(R.h1__edge_count(V.I1, V.I0)))) | [Aggregation.SUM, Combination.PRODUCT]
        R.h1 / 1 | [Transformation.IDENTITY]

    Setting :code:`normalize=False` drops the counting rules and leaves the plain walk-and-sum this module
    used to be - which is not what PyG computes. Two things to know about the edges either way, both shared
    with :class:`~neuralogic.nn.module.gnn.sg.SGConv`: an edge valued other than ``1.0`` scales the message
    it carries but does not enter the degree, which is a count rather than PyG's sum of edge weights, so the
    two agree exactly at ``1.0`` and nowhere else; and without self-loops a node of degree zero has no
    normalization atom at all, so its hop rules do not ground, where PyG makes that factor zero instead -
    agreeing on the value and not on whether the head exists.

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
        Default: ``2``
    activation : TransformationFunction
        Activation function of the output.
        Default: ``Transformation.IDENTITY``
    aggregation : AggregationFunction
        Aggregation function of nodes' neighbors.
        Default: ``Aggregation.SUM``
    normalize : bool
        Add symmetric normalization. No self-loops are added either way, matching PyG.
        Default: ``True``

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_name: str,
        feature_name: str,
        edge_name: str,
        k: int = 2,
        activation: TransformationFunction = Transformation.IDENTITY,
        aggregation: AggregationFunction = Aggregation.SUM,
        normalize: bool = True,
    ):
        self.output_name = output_name
        self.feature_name = feature_name
        self.edge_name = edge_name

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.k = k
        self.activation = activation
        self.aggregation = aggregation
        self.normalize = normalize

    def __call__(self):
        metadata = Metadata(aggregation=self.aggregation, combination=Combination.PRODUCT)
        head = R.get(self.output_name)
        feature = R.get(self.feature_name)
        edge = R.get(self.edge_name)
        edge_count = R.get(f"{self.output_name}__edge_count")

        normalization = []
        if self.normalize:
            count_metadata = Metadata(aggregation=Aggregation.COUNT)

            normalization = [
                (edge_count(V.I, V.J) <= edge(V.J, V.X)) | count_metadata,
                (edge_count(V.I, V.J) <= edge(V.I, V.X)) | count_metadata,
                edge_count / 2 | Metadata(combination=Combination.PRODUCT, transformation=Transformation.INVERSE),
            ]

        hop_rules = []

        for i in range(self.k + 1):
            body = [feature(f"I{i}")]
            for near, far in zip(range(i), range(1, i + 1)):
                body.append(edge(f"I{far}", f"I{near}"))
                if self.normalize:
                    body.append(Transformation.SQRT(edge_count(f"I{far}", f"I{near}")))

            hop_rules.append((head(V.I0)[self.out_channels, self.in_channels] <= body) | metadata)

        return [
            *normalization,
            *hop_rules,
            head / 1 | Metadata(transformation=self.activation),
        ]
