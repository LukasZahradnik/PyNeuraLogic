from typing import List, Optional

from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.enums import Activation, Aggregation
from neuralogic.core.constructs.factories import R, V
from neuralogic.utils.module.module import Module


class RGCNConv(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_name: str,
        feature_name: str,
        edge_name: Optional[str],
        relations: List[str],
        activation: Activation = Activation.IDENTITY,
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
        head = R.get(self.output_name)(V.X)
        metadata = Metadata(activation=Activation.IDENTITY, aggregation=self.aggregation)
        feature = R.get(self.feature_name)(V.Y)[self.out_channels, self.in_channels]

        if self.edge_name is not None:
            relation_rules = [
                ((head <= (feature, R.get(self.edge_name)(V.Y, relation, V.X))) | metadata)
                for relation in self.relations
            ]
        else:
            relation_rules = [
                ((head <= (feature, R.get(relation)(V.Y, V.X))) | metadata) for relation in self.relations
            ]

        return [
            (head <= R.get(self.feature_name)(V.X)[self.out_channels, self.in_channels]) | metadata,
            *relation_rules,
            R.get(self.output_name) / 1 | Metadata(activation=self.activation),
        ]
