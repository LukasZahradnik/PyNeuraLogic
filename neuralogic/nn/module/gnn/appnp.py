from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Transformation, Aggregation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class APPNPConv(Module):
    r"""
    Approximate Personalized Propagation of Neural Predictions layer from
    `"Predict then Propagate: Graph Neural Networks meet Personalized PageRank" <https://arxiv.org/abs/1810.05997>`_.
    Which can be expressed as:

    .. math::
        \mathbf{x}^{0}_i = \mathbf{x}_i

    .. math::
        \mathbf{x}^{k}_i = \alpha \cdot \mathbf{x}^0_i + (1 - \alpha) \cdot
        {agg}_{j \in \mathcal{N}(i)}(\mathbf{x}^{k - 1}_j)

    .. math::
        \mathbf{x}^{\prime}_i = act(\mathbf{x}^{K}_i)

    Where *act* is an activation function and *agg* aggregation function.


    The first part of the second equation that is ":math:`\alpha \cdot \mathbf{x}^0_i`" is expressed
    in the logic form as:

    .. code-block:: logtalk

        R.<output_name>__<k>(V.I) <= R.<feature_name>(V.I)[<alpha>].fixed()

    The second part of the second equation that is
    ":math:`(1 - \alpha) \cdot {agg}_{j \in \mathcal{N}(i)}(\mathbf{x}^{k - 1}_j)`" is expressed as:

    .. code-block:: logtalk

        R.<output_name>__<k>(V.I) <= (R.<output_name>__<k-1>(V.J)[1 - <alpha>].fixed(), R.<edge_name>(V.J, V.I))

    Examples
    --------

    The whole computation of this module
    (parametrized as :code:`APPNPConv("h1", "h0", "_edge", 3, 0.1, Transformation.SIGMOID)`) is as follows:

    .. code:: logtalk

        metadata = Metadata(transformation=Transformation.IDENTITY, aggregation=Aggregation.SUM)

        (R.h1__1(V.I) <= R.h0(V.I)[0.1].fixed()) | metadata
        (R.h1__1(V.I) <= (R.h0(V.J)[0.9].fixed(), R._edge(V.J, V.I))) | metadata
        R.h1__1/1 [Transformation.IDENTITY]

        (R.h1__2(V.I) <= <0.1> R.h0(V.I)) | metadata
        (R.h1__2(V.I) <= (<0.9> R.h1__1(V.J), R._edge(V.J, V.I))) | metadata
        R.h1__2/1 [Transformation.IDENTITY]

        (R.h1(V.I) <= <0.1> R.h0(V.I)) | metadata
        (R.h1(V.I) <= (<0.9> R.h1__2(V.J), R._edge(V.J, V.I))) | metadata
        R.h1 / 1 [Transformation.SIGMOID]


    Parameters
    ----------

    output_name : str
        Output (head) predicate name of the module.
    feature_name : str
        Feature predicate name to get features from.
    edge_name : str
        Edge predicate name to use for neighborhood relations.
    k : int
        Number of iterations
    alpha : float
        Teleport probability
    activation : Transformation
        Activation function of the output.
        Default: ``Transformation.IDENTITY``
    aggregation : Aggregation
        Aggregation function of nodes' neighbors.
        Default: ``Aggregation.SUM``

    """

    def __init__(
        self,
        output_name: str,
        feature_name: str,
        edge_name: str,
        k: int,
        alpha: float,
        activation: Transformation = Transformation.IDENTITY,
        aggregation: Aggregation = Aggregation.SUM,
    ):
        self.output_name = output_name
        self.feature_name = feature_name
        self.edge_name = edge_name

        self.alpha = alpha
        self.k = k

        self.aggregation = aggregation
        self.activation = activation

    def __call__(self):
        head = R.get(self.output_name)(V.I)
        metadata = Metadata(transformation=Transformation.IDENTITY, aggregation=self.aggregation)
        edge = R.get(self.edge_name)
        feature = R.get(self.feature_name)

        rules = []
        for k in range(1, self.k):
            k_head = R.get(f"{self.output_name}__{k}")(V.I)
            rules.append((k_head <= feature(V.I)[self.alpha].fixed()) | metadata)

            if k == 1:
                rules.append((k_head <= (feature(V.J)[1 - self.alpha].fixed(), edge(V.J, V.I))) | metadata)
            else:
                rules.append(
                    (k_head <= (R.get(f"{self.output_name}__{k - 1}")(V.J)[1 - self.alpha].fixed(), edge(V.J, V.I)))
                    | metadata
                )
            rules.append(R.get(f"{self.output_name}__{k}") / 1 | Metadata(transformation=Transformation.IDENTITY))

        if self.k == 1:
            output_rule = head <= (feature(V.J)[1 - self.alpha].fixed(), edge(V.J, V.I))
        else:
            output_rule = head <= (
                R.get(f"{self.output_name}__{self.k - 1}")(V.J)[1 - self.alpha].fixed(),
                edge(V.J, V.I),
            )

        return [
            *rules,
            (head <= feature(V.I)[self.alpha].fixed()) | metadata,
            output_rule | metadata,
            R.get(self.output_name) / 1 | Metadata(transformation=self.activation),
        ]
