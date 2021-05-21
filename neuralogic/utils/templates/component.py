from typing import List

from neuralogic.core import Template
from neuralogic.core.settings import Aggregation, Activation


class AbstractComponent:
    features_name = "node_features"

    def __init__(
        self,
        *,
        in_channels: int,
        activation: Activation = Activation.IDENTITY,
        aggregation: Aggregation = Aggregation.SUM,
        name=None,
        has_edge_attrs=True,
    ):
        self.in_channels = in_channels
        self.has_edge_attrs = has_edge_attrs
        self.aggregation = aggregation
        self.activation = activation
        self.name = name

    def build(self, template: Template, layer_count: int, previous_names: List[str], next_num_channels: int) -> str:
        pass
