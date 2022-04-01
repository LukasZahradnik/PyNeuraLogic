from typing import Union

from neuralogic.core.enums import Activation, Aggregation, ActivationAgg, ActivationAggregation


class Metadata:
    def __init__(
        self,
        offset=None,
        learnable: bool = None,
        activation: Union[str, Activation, ActivationAgg, ActivationAggregation] = None,
        aggregation: Aggregation = None,
        duplicit_grounding: bool = False,
    ):
        self.offset = offset
        self.learnable = learnable
        self.activation = activation
        self.aggregation = aggregation
        self.duplicit_grounding = duplicit_grounding

    def __str__(self):
        metadata_list = []
        if self.offset is not None:
            metadata_list.append(f"offset={self.offset}")
        if self.learnable is not None:
            metadata_list.append(f"learnable={str(self.learnable).lower()}")
        if self.activation is not None:
            metadata_list.append(f"activation={str(self.activation)}")
        if self.aggregation is not None:
            metadata_list.append(f"aggregation={self.aggregation.value}")
        return f"[{', '.join(metadata_list)}]"
