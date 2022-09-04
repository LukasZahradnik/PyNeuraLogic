from typing import Optional, List

from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Transformation, Aggregation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class MAGNNMean(Module):
    r"""
    Intra-metapath Aggregation module with Mean encoder from
    `"MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding" <https://arxiv.org/abs/2002.01680>`_.
    Which can be expressed as:

    .. math::

        \mathbf{h}_{P(v,u)} = MEAN(\{\mathbf{x}_t |  \forall t \in P(v,u) \})

    .. math::

        \mathbf{h}^P_{v} = act(\sum_{u \in N^P_v} \mathbf{h}_{P(v,u)})

    Where *act* is an activation function, :math:`P(v,u)` is a single metapath instance, :math:`N^P_{v}` is set of
    metapath-based neighbors.

    Parameters
    ----------

    output_name : str
        Output (head) predicate name of the module.
    feature_name : str
        Feature predicate name to get features from.
    relation_name : str
        Relation predicate name for connectivity checks between entities.
    type_name : Optional[str]
        Metapath type predicate name. If none, ``meta_paths`` will be used instead.
    meta_paths : List[str]
        Name of types forming a single metapath.
    activation : Transformation
        Activation function of the output.
        Default: ``Transformation.SIGMOID``
    """

    def __init__(
        self,
        output_name: str,
        feature_name: str,
        relation_name: str,
        type_name: Optional[str],
        meta_paths: List[str],
        activation: Transformation = Transformation.SIGMOID,
        aggregation: Aggregation = Aggregation.SUM,
    ):
        self.output_name = output_name
        self.feature_name = feature_name
        self.relation_name = relation_name
        self.type_name = type_name
        self.meta_paths = meta_paths

        self.aggregation = aggregation
        self.activation = activation

    def __call__(self):
        metadata = Metadata(duplicit_grounding=True, transformation=Transformation.IDENTITY)
        length = len(self.meta_paths)
        feature = R.get(self.feature_name)
        relation = R.get(self.relation_name)

        meta_paths = [relation(f"V{index}", f"V{index + 1}") for index in range(len(self.meta_paths) - 1)]

        if self.type_name is None:
            meta_paths.extend(R.get(type)(f"V{index}") for index, type in enumerate(self.meta_paths))
        else:
            type_relation = R.get(self.type_name)
            meta_paths.extend(type_relation(f"V{index}", type) for index, type in enumerate(self.meta_paths))

        if length == 0:
            meta_paths.extend(feature(f"V{index}") for index in range(len(self.meta_paths)))
        else:
            meta_paths.extend(feature(f"V{index}")[1 / length].fixed() for index in range(len(self.meta_paths)))

        return [
            (R.get(self.output_name)(V.V0) <= meta_paths) | metadata,
            R.get(self.output_name) / 1 | Metadata(transformation=self.activation),
        ]


class MAGNNLinear(MAGNNMean):
    r"""
    Intra-metapath Aggregation module with Linear encoder from
    `"MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding" <https://arxiv.org/abs/2002.01680>`_.
    Which can be expressed as:

    .. math::

        \mathbf{h}_{P(v,u)} = \mathbf{W}_p \cdot MEAN(\{\mathbf{x}_t |  \forall t \in P(v,u) \})

    .. math::

        \mathbf{h}^P_{v} = act(\sum_{u \in N^P_v} \mathbf{h}_{P(v,u)})

    Where *act* is an activation function, :math:`P(v,u)` is a single metapath instance, :math:`N^P_{v}` is set of
    metapath-based neighbors.

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
    relation_name : str
        Relation predicate name for connectivity checks between entities.
    type_name : Optional[str]
        Metapath type predicate name. If none, ``meta_paths`` will be used instead.
    meta_paths : List[str]
        Name of types forming a single metapath.
    activation : Transformation
        Activation function of the output.
        Default: ``Transformation.SIGMOID``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_name: str,
        feature_name: str,
        relation_name: str,
        type_name: Optional[str],
        meta_paths: List[str],
        activation: Transformation = Transformation.SIGMOID,
        aggregation: Aggregation = Aggregation.SUM,
    ):
        super().__init__(output_name, feature_name, relation_name, type_name, meta_paths, activation, aggregation)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __call__(self):
        rules = super().__call__()

        rules[0].head = rules[0].head[self.out_channels, self.in_channels]
        return rules
