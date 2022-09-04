from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Transformation, Aggregation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class SAGEConv(Module):
    r"""
    GraphSAGE layer from `"Inductive Representation Learning on Large Graphs" <https://arxiv.org/abs/1706.02216>`_.
    Which can be expressed as:

    .. math::
        \mathbf{x}^{\prime}_i = act(\mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
         {agg}_{j \in \mathcal{N}(i)}(\mathbf{x}_j)))

    Where *act* is an activation function, *agg* aggregation function and *W*'s are learnable parameters.
    This equation is translated into the logic form as:

    .. code:: logtalk

         (R.<output_name>(V.I)[<W1>] <= (R.<feature_name>(V.J), R.<edge_name>(V.J, V.I))) | [<aggregation>, Transformation.IDENTITY]
         (R.<output_name>(V.I)[<W2>] <= R.<feature_name>(V.I)) | [Transformation.IDENTITY]
         R.<output_name> / 1 | [<activation>]

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
        Default: ``Aggregation.AVG``

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_name: str,
        feature_name: str,
        edge_name: str,
        activation: Transformation = Transformation.IDENTITY,
        aggregation: Aggregation = Aggregation.AVG,
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
        metadata = Metadata(transformation=Transformation.IDENTITY, aggregation=self.aggregation)

        return [
            (head <= (R.get(self.feature_name)(V.J), R.get(self.edge_name)(V.J, V.I))) | metadata,
            (head <= R.get(self.feature_name)(V.I)) | metadata,
            R.get(self.output_name) / 1 | Metadata(transformation=self.activation),
        ]
