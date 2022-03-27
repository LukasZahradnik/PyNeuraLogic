from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.enums import Activation, Aggregation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class GCNConv(Module):
    r"""
    Graph Convolutional layer from `"Semi-supervised Classification with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_.
    Which can be expressed as:

    .. math::
        \mathbf{x}^{\prime}_i = act(\mathbf{W} \cdot {agg}_{j \in \mathcal{N}(i)}(\mathbf{x}_j))

    Where *act* is an activation function, *agg* aggregation function and *W* is a learnable parameter. This equation is
    translated into the logic form as:

    .. code:: logtalk

         (R.<output_name>(V.I)[<W>] <= (R.<feature_name>(V.J), R.<edge_name>(V.J, V.I))) | [<aggregation>, Activation.IDENTITY]
         R.<output_name> / 1 | [<activation>]

    Examples
    --------

    The whole computation of this module (parametrized as :code:`GCNConv(2, 3, "h1", "h0", "_edge")`) is as follows:

    .. code:: logtalk

        (R.h1(V.I)[2, 3] <= (R.h0(V.J), R._edge(V.J, V.I)) | [Aggregation.SUM, Activation.IDENTITY]
        R.h1 / 1 | [Activation.IDENTITY]

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
    activation : Activation
        Activation function of the output.
        Default: ``Activation.IDENTITY``
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
        activation: Activation = Activation.IDENTITY,
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
        metadata = Metadata(activation=Activation.IDENTITY, aggregation=self.aggregation)

        return [
            (head <= (R.get(self.feature_name)(V.J), R.get(self.edge_name)(V.J, V.I))) | metadata,
            R.get(self.output_name) / 1 | Metadata(activation=self.activation),
        ]
