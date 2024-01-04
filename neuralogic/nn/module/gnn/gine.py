from typing import Optional

from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Transformation, Aggregation, Combination
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class GINEConv(Module):
    r"""
    GINEConv layer from `"Strategies for Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_.

    Parameters
    ----------

    in_channels : int
        Input feature size.
    out_channels : int
        Output feature size.
    feature_name : str
        Feature predicate name to get features from.
    edge_name : str
        Edge predicate name to use for neighborhood relations.
    nn_name : str
        Neural network predicate name.
    eps : float
        :math:`\epsilon`-value.
        Default: ``0.0``
    train_eps : bool
        Is ``eps`` trainable parameter.
        Default: ``false``
    edge_dim : Optional[int]
        Dimension of edge features (``None`` is projection to ``in_channels`` is not needed).
        Default: ``None``
    """

    def __init__(
        self,
        in_channels: int,
        feature_name: str,
        edge_name: str,
        nn_name: str,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
    ):
        self.feature_name = feature_name
        self.edge_name = edge_name
        self.nn_name = nn_name

        self.eps = eps
        self.train_eps = train_eps

        self.in_channels = in_channels
        self.edge_dim: Optional[int] = edge_dim

    def __call__(self):
        feat_sum = R.get(f"{self.nn_name}__gine_feat_sum")
        feat_agg = R.get(f"{self.nn_name}__gine_feat_agg")
        out = R.get(f"{self.nn_name}__gine_out")
        x = R.get(self.feature_name)
        e = R.get(self.edge_name)

        x_eps = x(V.I)[self.eps + 1]
        if not self.train_eps:
            x_eps = x_eps.fixed()

        e_proj = []
        if self.edge_dim is not None:
            e = R.get(f"{self.nn_name}__gine_edge_proj")
            e_proj = [
                (e(V.I, V.J)[self.in_channels, self.edge_dim] <= R.get(self.edge_name)(V.I, V.J))
                | Metadata(transformation=Transformation.IDENTITY),
                e / 2 | Metadata(transformation=Transformation.IDENTITY),
            ]

        return [
            *e_proj,
            (feat_sum(V.I, V.J) <= (x(V.J), e(V.J, V.I)))
            | Metadata(transformation=Transformation.RELU, combination=Combination.SUM),
            feat_sum / 2 | Metadata(transformation=Transformation.IDENTITY),
            (feat_agg(V.I) <= feat_sum(V.I, V.J))
            | Metadata(transformation=Transformation.IDENTITY, aggregation=Aggregation.SUM),
            feat_agg / 1 | Metadata(transformation=Transformation.IDENTITY),
            (out(V.I) <= (x_eps, feat_agg(V.I)))
            | Metadata(transformation=Transformation.IDENTITY, combination=Combination.SUM),
            out / 1 | Metadata(transformation=Transformation.IDENTITY),
            (R.get(self.nn_name)(V.I) <= out(V.I)) | Metadata(transformation=Transformation.IDENTITY),
            R.get(self.nn_name) / 1 | Metadata(transformation=Transformation.IDENTITY),
        ]
