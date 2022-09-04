from typing import List, Optional

from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Transformation, Aggregation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class MetaConv(Module):
    r"""
    Metagraph Convolutional Unit layer from
    `Meta-GNN: metagraph neural network for semi-supervised learning in attributed heterogeneous information networks <https://dl.acm.org/doi/10.1145/3341161.3342859>`_.
    Which can be expressed as:

    .. math::
        \mathbf{x}^{\prime}_i = act(\mathbf{W_0} \cdot \mathbf{x}_i + {agg}_{j \in \mathcal{N}_r(i)}
        \sum_{k \in \mathcal{K}}
        (\mathbf{W_k} \cdot \mathbf{x}_j))

    Where *act* is an activation function, *agg* aggregation function (by default average), :math:`W_0` is a learnable
    root parameter and :math:`W_k` is a learnable parameter for each role.

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
    role_name : Optional[str]
        Role predicate name to use for role relations. When :code:`None`, elements from :code:`roles` are used instead.
    roles : List[str]
        List of relations' names
    activation : Transformation
        Activation function of the output.
        Default: ``Transformation.SIGMOID``
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
        role_name: Optional[str],
        roles: List[str],
        activation: Transformation = Transformation.SIGMOID,
        aggregation: Aggregation = Aggregation.AVG,
    ):
        self.output_name = output_name
        self.feature_name = feature_name
        self.role_name = role_name

        self.roles = roles

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.aggregation = aggregation
        self.activation = activation

    def __call__(self):
        head = R.get(self.output_name)(V.I)
        role_head = R.get(f"{self.output_name}__roles")

        metadata = Metadata(transformation=Transformation.IDENTITY, aggregation=self.aggregation)
        feature = R.get(self.feature_name)(V.J)[self.out_channels, self.in_channels]

        if self.role_name is not None:
            role_rules = [
                ((role_head(V.I, role) <= (feature, R.get(self.role_name)(V.J, role, V.I))) | [Transformation.IDENTITY])
                for role in self.roles
            ]
        else:
            role_rules = [
                ((role_head(V.I, role) <= (feature, R.get(role)(V.J, V.I))) | [Transformation.IDENTITY])
                for role in self.roles
            ]

        return [
            (head <= role_head(V.I, V.R)) | metadata,
            (head <= R.get(self.feature_name)(V.I)[self.out_channels, self.in_channels]) | metadata,
            *role_rules,
            R.get(self.output_name) / 1 | Metadata(transformation=self.activation),
            role_head / 2 | [Transformation.IDENTITY],
        ]
