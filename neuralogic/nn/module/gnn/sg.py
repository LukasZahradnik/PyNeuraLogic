from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Transformation, Aggregation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class SGConv(Module):
    r"""
    Simple Graph Convolutional layer from `"Simplifying Graph Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_.
    Which can be expressed as:

    .. math::
        \mathbf{x}^{\prime}_i = act(\mathbf{W} \cdot {agg}_{j \in \mathcal{N}^k(i)}(\mathbf{x}_j))

    Where *act* is an activation function, *agg* aggregation function, *W* is a learnable parameter
    and :math:`\mathcal{N}^k(i)` denotes nodes that are *k* hops away from the node *i*.
    This equation is translated into the logic form as:

    .. code:: logtalk

        (R.<output_name>(V.I)[<W>] <= (
            R.<feature_name>(V.I<k>),
            R.<edge_name>(V.I<1>, V.I<0>), R.<edge_name>(V.I<2>, V.I<1>), ..., R.<edge_name>(V.I<k>, V.I<k-1>),
        )) | [<aggregation>, Transformation.IDENTITY]

        R.<output_name> / 1 | [<activation>]

    Examples
    --------

    The whole computation of this module (parametrized as :code:`SGConv(2, 3, "h1", "h0", "_edge", 2)`) is as follows:

    .. code:: logtalk

        (R.h1(V.I0)[3, 2] <= (R.h0(V.I2), R._edge(V.I1, V.I0), R._edge(V.I2, V.I1))) | [Transformation.IDENTITY, Aggregation.SUM]
        R.h1 / 1 | [Transformation.IDENTITY]

    Module parametrized as :code:`SGConv(2, 3, "h1", "h0", "_edge", 1)` translates into:

    .. code:: logtalk

        (R.h1(V.I0)[3, 2] <= (R.h0(V.I1), R._edge(V.I1, V.I0))) | [Transformation.IDENTITY, Aggregation.SUM]
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
        Default: ``1``
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
        k: int = 1,
        activation: Transformation = Transformation.IDENTITY,
        aggregation: Aggregation = Aggregation.SUM,
    ):
        self.output_name = output_name
        self.feature_name = feature_name
        self.edge_name = edge_name

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.k = k

        self.aggregation = aggregation
        self.activation = activation

    def __call__(self):
        head = R.get(self.output_name)(V.I0)[self.out_channels, self.in_channels]
        metadata = Metadata(
            transformation=Transformation.IDENTITY, aggregation=self.aggregation, duplicit_grounding=True
        )
        edge = R.get(self.edge_name)
        feature = R.get(self.feature_name)

        return [
            (
                head
                <= (
                    feature(f"I{self.k}"),
                    *(edge(f"I{b}", f"I{a}") for a, b in zip(range(self.k), range(1, self.k + 1))),
                )
            )
            | metadata,
            R.get(self.output_name) / 1 | Metadata(transformation=self.activation),
        ]
