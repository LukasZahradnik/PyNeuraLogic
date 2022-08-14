from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Transformation, Aggregation, Combination
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class ResGatedGraphConv(Module):
    r"""
    Residual Gated Graph Convolutional layer from `"Residual Gated Graph ConvNets" <https://arxiv.org/abs/1711.07553>`_.
    Which can be expressed as:

    .. math::

        \mathbf{x}^{\prime}_i = act(\mathbf{W}_1 \mathbf{x}_i +
         {agg}_{j \in \mathcal{N}(i)}(\eta_{i,j} \odot \mathbf{W}_2 \mathbf{x}_j))

    .. math::

        \mathbf{\eta}_{i,j} = gating\_act(\mathbf{W}_3 \mathbf{x}_i + \mathbf{W}_4 \mathbf{x}_j)

    Where *act* is an activation function, *agg* aggregation function, *gating_act* is a gating activation function and
    :math:`W_n` are learnable parameters. This equation is translated into the logic form as:

    .. code:: logtalk

        (R.<output_name>__gate(V.I, V.J) <= (R.<feature_name>(V.I)[<W>], R.<feature_name>(V.J)[<W>])) | [Transformation.IDENTITY]
        R.<output_name>__gate / 2 | [<activation>]

        (R.<output_name>(V.I) <= R.<feature_name>(V.I)[<W>]) | [Transformation.IDENTITY]
        (R.<output_name>(V.I) <= (
            R.<output_name>__gate(V.I, V.J), R.<feature_name>(V.J)[<W>], R.<edge_name>(V.J, V.I))
        ) | Metadata(activation="elementproduct-identity", aggregation=<aggregation>)

        R.<output_name> / 1 | [<activation>]

    Examples
    --------

    The whole computation of this module (parametrized as :code:`ResGatedGraphConv(1, 2, "h1", "h0", "_edge")`)
    is as follows:

    .. code:: logtalk

        metadata = Metadata(activation="elementproduct-identity", aggregation=Aggregation.SUM)

        (R.h1__gate(V.I, V.J) <= (R.h0(V.I)[2, 1], R.h0(V.J)[2, 1])) | [Transformation.IDENTITY]
        R.h1__gate / 2 | [Transformation.SIGMOID]

        (R.h1(V.I) <= R.h0(V.I)[2, 1]) | [Transformation.IDENTITY]
        (R.h1(V.I) <= (R.h1__gate(V.I, V.J), R.h0(V.J)[2, 1], R._edge(V.J, V.I))) | metadata
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
    gating_activation : Transformation
        Gating activation function.
        Default: ``Transformation.SIGMOID``
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
        gating_activation: Transformation = Transformation.SIGMOID,
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
        self.gating_activation = gating_activation

    def __call__(self):
        head = R.get(self.output_name)(V.I)
        feature = R.get(self.feature_name)
        gate = R.get(f"{self.output_name}__gate")

        w = self.out_channels, self.in_channels
        prod_metadata = Metadata(
            combination=Combination.ELPRODUCT, transformation=Transformation.IDENTITY, aggregation=self.aggregation
        )

        return [
            (gate(V.I, V.J) <= (feature(V.I)[w], feature(V.J)[w])) | [Transformation.IDENTITY],
            gate / 2 | Metadata(transformation=self.gating_activation),
            (head <= feature(V.I)[w]) | [Transformation.IDENTITY],
            (head <= (gate(V.I, V.J), feature(V.J)[w], R.get(self.edge_name)(V.J, V.I))) | prod_metadata,
            R.get(self.output_name) / 1 | Metadata(transformation=self.activation),
        ]
