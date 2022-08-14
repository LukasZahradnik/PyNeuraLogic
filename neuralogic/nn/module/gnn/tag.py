from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Transformation, Aggregation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class TAGConv(Module):
    r"""
    Topology Adaptive Graph Convolutional layer from
    `"Topology Adaptive Graph Convolutional Networks" <https://arxiv.org/abs/1710.10370>`_.
    Which can be expressed as:

    .. math::
        \mathbf{x}^{\prime}_i = act(\sum_{k=0}^K \mathbf{W}_k \cdot {agg}_{j \in \mathcal{N}^k(i)}(\mathbf{x}_j))

    Where *act* is an activation function, *agg* aggregation function, *Wk* are learnable parameters and
    :math:`\mathcal{N}^k(i)` denotes nodes that are *k* hops away from the node *i*. This equation is translated into
    the logic form as:

    This equation is translated into the logic form as:

    .. code:: logtalk

        (R.<output_name>(V.I0)[<W0>] <= R.<feature_name>(V.I0)) | [<aggregation>, Transformation.IDENTITY]
        (R.<output_name>(V.I0)[<W1>] <= (R.<feature_name>(V.I1), R.<edge_name>(V.I1, V.I0))) | [<aggregation>, Transformation.IDENTITY]
        (R.<output_name>(V.I0)[<W2>] <= (R.<feature_name>(V.I2), R.<edge_name>(V.I1, V.I0), R.<edge_name>(V.I2, V.I1)) | [<aggregation>, Transformation.IDENTITY]
        ...
        (R.<output_name>(V.I0)[<Wk>] <= (R.<feature_name>(V.I<k>), R.<edge_name>(V.I1, V.I0), ..., R.<edge_name>(V.I<k>, V.I<k-1>)) | [<aggregation>, Transformation.IDENTITY]
        R.<output_name> / 1 | [<activation>]

    Examples
    --------

    The whole computation of this module (parametrized as :code:`TAGConv(1, 2, "h1", "h0", "_edge")`) is as follows:

    .. code:: logtalk

        (R.h1(V.I0)[2, 2] <= R.h0(V.I0)) | [Aggregation.SUM, Transformation.IDENTITY]
        (R.h1(V.I0)[2, 1] <= (R.h0(V.I1), R._edge(V.I1, V.I0)) | [Aggregation.SUM, Transformation.IDENTITY]
        (R.h1(V.I0)[2, 1] <= (R.h0(V.I2), R._edge(V.I1, V.I0), R._edge(V.I2, V.I1)) | [Aggregation.SUM, Transformation.IDENTITY]
        R.h1 / 1 | [Transformation.IDENTITY]

    Module parametrized as :code:`TAGConv(1, 2, "h1", "h0", "_edge", 1)` translates into:

    .. code:: logtalk

        (R.h1(V.I0)[2, 1] <= R.h0(V.I0)) | [Aggregation.SUM, Transformation.IDENTITY]
        (R.h1(V.I0)[2, 1] <= (R.h0(V.I1), R._edge(V.I1, V.I0)) | [Aggregation.SUM, Transformation.IDENTITY]
        R.h1 / 1 | [Transformation.IDENTITY]

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
        k: int = 2,
        activation: Transformation = Transformation.IDENTITY,
        aggregation: Aggregation = Aggregation.SUM,
    ):
        self.output_name = output_name
        self.feature_name = feature_name
        self.edge_name = edge_name

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.k = k
        self.activation = activation
        self.aggregation = aggregation

    def __call__(self):
        metadata = Metadata(transformation=Transformation.IDENTITY, aggregation=self.aggregation)
        head = R.get(self.output_name)
        feature = R.get(self.feature_name)
        edge = R.get(self.edge_name)

        hop_rules = []

        for i in range(self.k + 1):
            hop_rules.append(
                (
                    head(V.I0)[self.out_channels, self.in_channels]
                    <= (
                        feature(f"I{i}"),
                        *(edge(f"I{b}", f"I{a}") for a, b in zip(range(i), range(1, i + 1))),
                    )
                )
                | metadata
            )

        return [
            *hop_rules,
            head / 1 | Metadata(transformation=self.activation),
        ]
