from typing import Optional

from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Transformation, Aggregation, Combination
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.general.mlp import MLP
from neuralogic.nn.module.module import Module


class GENConv(Module):
    r"""
    GENConv layer from `"DeeperGCN: All You Need to Train Deeper GCNs" <https://arxiv.org/abs/2006.07739>`_.

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
    aggregation : Aggregation
        The aggregation function.
        Default: ``Aggregation.SOFTMAX``
    num_layers : int
        The number of MLP layers.
        Default: ``2``
    expansion : int
        The expansion factor of hidden channels in MLP.
        Default: ``2``
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
        out_channels: int,
        output_name: str,
        feature_name: str,
        edge_name: str,
        aggregation: Aggregation = Aggregation.SOFTMAX,
        num_layers: int = 2,
        expansion: int = 2,
        eps: float = 1e-7,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
    ):
        self.output_name = output_name
        self.feature_name = feature_name
        self.edge_name = edge_name

        self.aggregation = aggregation
        self.num_layers = num_layers
        self.expansion = expansion

        self.eps = eps
        self.train_eps = train_eps

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim: Optional[int] = edge_dim

    def __call__(self):
        feat_sum = R.get(f"{self.output_name}__gen_feat_sum")
        feat_agg = R.get(f"{self.output_name}__gen_feat_agg")
        out = R.get(f"{self.output_name}__gen_out")
        x = R.get(self.feature_name)
        e = R.get(self.edge_name)

        eps = R.get(f"{self.output_name}__eps")
        v_eps = eps[self.eps]
        if not self.train_eps:
            v_eps = v_eps.fixed()

        e_proj = []
        if self.edge_dim is not None and self.out_channels != self.edge_dim:
            e = R.get(f"{self.output_name}__gen_edge_proj")
            e_proj = [
                (e(V.I, V.J)[self.out_channels, self.edge_dim] <= R.get(self.edge_name)(V.I, V.J))
                | Metadata(transformation=Transformation.IDENTITY),
                e / 2 | Metadata(transformation=Transformation.IDENTITY),
            ]

        channels = [self.out_channels]
        for _ in range(self.num_layers - 1):
            channels.append(self.out_channels * self.expansion)
        channels.append(self.out_channels)

        mlp = MLP(channels, self.output_name, f"{self.output_name}__gen_out", Transformation.IDENTITY)

        j_feat = x(V.J)
        i_feat = x(V.I)
        if self.in_channels != self.out_channels:
            j_feat = x(V.J)[self.out_channels, self.in_channels]
            i_feat = x(V.I)[self.out_channels, self.in_channels]

        return [
            v_eps,
            *e_proj,
            (feat_sum(V.I, V.J) <= (j_feat, e(V.J, V.I)))
            | Metadata(transformation=Transformation.RELU, combination=Combination.SUM),
            feat_sum / 2 | Metadata(transformation=Transformation.IDENTITY),
            (feat_agg(V.I) <= (feat_sum(V.I, V.J), eps))
            | Metadata(
                transformation=Transformation.IDENTITY, aggregation=self.aggregation, combination=Combination.SUM
            ),
            feat_agg / 1 | Metadata(transformation=Transformation.IDENTITY),
            (out(V.I) <= (i_feat, feat_agg(V.I)))
            | Metadata(transformation=Transformation.IDENTITY, combination=Combination.SUM),
            out / 1 | Metadata(transformation=Transformation.IDENTITY),
            *mlp(),
        ]
