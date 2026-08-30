from neuralogic.core.constructs.factories import R, V
from neuralogic.core.constructs.function import Aggregation, Combination, Transformation
from neuralogic.core.constructs.function.function import TransformationFunction
from neuralogic.core.constructs.metadata import Metadata
from neuralogic.nn.module.module import Module


class GATv2Conv(Module):
    r"""
    GATv2 layer from `"How Attentive are Graph Attention Networks?" <https://arxiv.org/abs/2105.14491>`_.

    .. math::
        \alpha_{ij} = \mathrm{softmax}_{j \in \mathcal{N}(i) \cup \{i\}}
        \left( \mathbf{a}^{\top} \mathrm{LeakyReLU}
        \left( \mathbf{W}_{l} \mathbf{x}_{i} + \mathbf{W}_{r} \mathbf{x}_{j} \right) \right)

    .. math::
        \mathbf{x}^{\prime}_{i} = act \left( \sum_{j \in \mathcal{N}(i) \cup \{i\}}
        \alpha_{ij} \mathbf{W}_{r} \mathbf{x}_{j} \right)

    The score is one *number* per edge, so :math:`\mathbf{a}` is a ``1 x out_channels`` weight and it sits on
    the **body** of the rule that normalizes it, not on the head: a head weight is applied by the atom neuron
    *after* the aggregation, and the softmax aggregation casts every input it is given to a scalar. The
    normalization is `Aggregation.SOFTMAX` with ``agg_terms``, which groups the groundings by the terms it is
    not given - so over the neighbours ``J`` for each target ``I``, while staying one value per edge. That is
    the same shape :class:`~neuralogic.nn.module.general.attention.Attention` uses.

    The edge appears as a *hidden* atom in both rules, and only to restrict the grounding: without it the
    score would ground for every pair of nodes and the softmax would normalize over the whole graph rather
    than over a neighbourhood. PyG's GATv2 takes edge *features* through a projection rather than an edge
    weight, so there is nothing for the edge's own value to mean here.

    The slope defaults to PyG's ``0.2``, not to the backend's own ``0.01``: this module exists to be PyG's
    layer, and quietly using a different slope made every comparison bend PyG instead. It is a per-rule
    slope now, so ``negative_slope`` says what it says.

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
    share_weights : bool
        Share weights in attention. Default: ``False``
    activation : TransformationFunction
        Activation function of the output.
        Default: ``Transformation.IDENTITY``
    add_self_loops : bool
        Let a node attend to itself, as PyG does. Default: ``True``
    negative_slope : float
        The LeakyReLU slope of the attention score. Default: ``0.2``, as in PyG.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_name: str,
        feature_name: str,
        edge_name: str,
        share_weights: bool = False,
        activation: TransformationFunction = Transformation.IDENTITY,
        add_self_loops: bool = True,
        negative_slope: float = 0.2,
    ):
        self.output_name = output_name
        self.feature_name = feature_name
        self.edge_name = edge_name

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.share_weights = share_weights
        self.activation = activation
        self.add_self_loops = add_self_loops
        self.negative_slope = negative_slope

    def __call__(self):
        w1 = f"{self.output_name}__right"
        w2 = w1 if self.share_weights else f"{self.output_name}__left"

        score = R.get(f"{self.output_name}__score")
        attention = R.get(f"{self.output_name}__attention")

        head = R.get(self.output_name)
        feature = R.get(self.feature_name)

        self_loops = []
        if self.add_self_loops:
            loop = R.get(f"{self.output_name}__edge")
            self_loops = [
                loop(V.I, V.I)[1.0].fixed(),
                loop(V.I, V.J) <= (R.get(self.edge_name)(V.I, V.J)),
            ]
            edge = R.hidden.get(f"{self.output_name}__edge")
        else:
            edge = R.hidden.get(self.edge_name)

        return [
            *self_loops,
            # the score per edge: leaky relu of the two endpoints' projections summed. The hidden edge is
            # there only to restrict the grounding to the neighbourhood - without it this grounds for every
            # pair of nodes, and the softmax below then normalizes over the whole graph
            (
                score(V.I, V.J)
                <= (
                    feature(V.I)[w2 : self.out_channels, self.in_channels],
                    feature(V.J)[w1 : self.out_channels, self.in_channels],
                    edge(V.J, V.I),
                )
            )
            | Metadata(
                combination=Combination.SUM,
                transformation=Transformation.LEAKY_RELU(self.negative_slope),
            ),
            # `a` on the body rather than the head: a head weight applies after the aggregation, and the
            # softmax aggregation casts every input to a scalar, so it has to be one number by then
            (attention(V.I, V.J) <= score(V.I, V.J)[f"{self.output_name}__att" : 1, self.out_channels])
            | Metadata(
                aggregation=Aggregation.SOFTMAX(agg_terms=["J"]),
                transformation=Transformation.IDENTITY,
            ),
            (
                head(V.I)
                <= (
                    attention(V.I, V.J),
                    feature(V.J)[w1 : self.out_channels, self.in_channels],
                    edge(V.J, V.I),
                )
            )
            | Metadata(aggregation=Aggregation.SUM, combination=Combination.PRODUCT),
            head / 1 | Metadata(transformation=self.activation),
        ]
