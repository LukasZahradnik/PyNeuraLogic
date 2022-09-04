from typing import List, Optional

from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Transformation, Aggregation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class RGCNConv(Module):
    r"""
    Relational Graph Convolutional layer from
    `Modeling Relational Data with Graph Convolutional Networks <https://arxiv.org/abs/1703.06103>`_.
    Which can be expressed as:

    .. math::
        \mathbf{x}^{\prime}_i = act(\mathbf{W_0} \cdot \mathbf{x}_i + \sum_{r \in \mathcal{R}}
        {agg}_{j \in \mathcal{N}_r(i)}(\mathbf{W_r} \cdot \mathbf{x}_j))

    Where *act* is an activation function, *agg* aggregation function (by default average), :math:`W_0` is a
    learnable root parameter and :math:`W_r` is a learnable parameter for each relation.

    The first part of the equation that is ":math:`\mathbf{W_0} \cdot \mathbf{x}_i`" can be expressed
    in the logic form as:

    .. code-block:: logtalk

        R.<output_name>(V.I) <= R.<feature_name>(V.I)[<W0>]

    Another part of the equation that is ":math:`{agg}_{j \in \mathcal{N}_r(i)}(\mathbf{W_r} \cdot \mathbf{x}_j)`"
    can be expressed as:

    .. code-block:: logtalk

        R.<output_name>(V.I) <= (R.<feature_name>(V.J)[<Wr>], R.<edge_name>(V.J, relation, V.I))

    where "relation" is a constant name, or as:

    .. code-block:: logtalk

        R.<output_name>(V.I) <= (R.<feature_name>(V.J)[<Wr>], R.<relation>(V.J, V.I))

    The outer summation, together with summing it with the first part, is handled by aggregation of all rules with the
    same head (and substitution).

    Examples
    --------

    The whole computation of this module
    (parametrized as :code:`RGCNConv(1, 2, "h1", "h0", "_edge", ["sibling", "parent"])`) is as follows:

    .. code:: logtalk

        metadata = Metadata(activation=Transformation.IDENTITY, aggregation=Aggregation.AVG)

        (R.h1(V.I) <= R.h0(V.I)[2, 1]) | metadata
        (R.h1(V.I) <= (R.h0(V.J)[2, 1], R._edge(V.J, sibling, V.I))) | metadata
        (R.h1(V.I) <= (R.h0(V.J)[2, 1], R._edge(V.J, parent, V.I))) | metadata
        R.h1 / 1 [Transformation.IDENTITY]

    Module parametrized as :code:`RGCNConv(1, 2, "h1", "h0", None, ["sibling", "parent"])` translates into:

    .. code:: logtalk

        metadata = Metadata(activation=Transformation.IDENTITY, aggregation=Aggregation.AVG)

        (R.h1(V.I) <= R.h0(V.I)[2, 1]) | metadata
        (R.h1(V.I) <= (R.h0(V.J)[2, 1], R.sibling(V.J, V.I))) | metadata
        (R.h1(V.I) <= (R.h0(V.J)[2, 1], R.parent(V.J, V.I))) | metadata
        R.h1 / 1 [Transformation.IDENTITY]

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
    edge_name : Optional[str]
        Edge predicate name to use for neighborhood relations. When :code:`None`, elements from :code:`relations`
        are used instead.
    relations : List[str]
        List of relations' names
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
        edge_name: Optional[str],
        relations: List[str],
        activation: Transformation = Transformation.IDENTITY,
        aggregation: Aggregation = Aggregation.AVG,
    ):
        self.output_name = output_name
        self.feature_name = feature_name
        self.edge_name = edge_name

        self.relations = relations

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.aggregation = aggregation
        self.activation = activation

    def __call__(self):
        head = R.get(self.output_name)(V.I)
        metadata = Metadata(transformation=Transformation.IDENTITY, aggregation=self.aggregation)
        feature = R.get(self.feature_name)(V.J)[self.out_channels, self.in_channels]

        if self.edge_name is not None:
            relation_rules = [
                ((head <= (feature, R.get(self.edge_name)(V.J, relation, V.I))) | metadata)
                for relation in self.relations
            ]
        else:
            relation_rules = [
                ((head <= (feature, R.get(relation)(V.J, V.I))) | metadata) for relation in self.relations
            ]

        return [
            (head <= R.get(self.feature_name)(V.I)[self.out_channels, self.in_channels]) | metadata,
            *relation_rules,
            R.get(self.output_name) / 1 | Metadata(transformation=self.activation),
        ]
