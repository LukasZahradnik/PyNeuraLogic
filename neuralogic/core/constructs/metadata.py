from typing import Union, Iterable, Callable

from neuralogic.core.constructs.function import Activation, ActivationAgg, Aggregation, Function


class Metadata:
    __slots__ = "offset", "learnable", "activation", "aggregation", "duplicit_grounding"

    def __init__(
        self,
        offset=None,
        learnable: bool = None,
        activation: Union[str, Activation, ActivationAgg] = None,
        aggregation: Union[str, Aggregation] = None,
        duplicit_grounding: bool = False,
    ):
        self.offset = offset
        self.learnable = learnable
        self.activation = activation
        self.aggregation = aggregation
        self.duplicit_grounding = duplicit_grounding

    @staticmethod
    def from_iterable(iterable: Iterable) -> "Metadata":
        metadata = Metadata()

        for entry in iterable:
            if isinstance(entry, Callable) and not isinstance(entry, Function):
                entry = entry()
            if isinstance(entry, Aggregation):
                metadata.aggregation = entry
            elif isinstance(entry, (Activation, ActivationAgg)):
                metadata.activation = entry
            else:
                raise NotImplementedError
        return metadata

    def __str__(self):
        metadata_list = []
        if self.offset is not None:
            metadata_list.append(f"offset={self.offset}")
        if self.learnable is not None:
            metadata_list.append(f"learnable={str(self.learnable).lower()}")
        if self.activation is not None:
            metadata_list.append(f"activation={str(self.activation)}")
        if self.aggregation is not None:
            metadata_list.append(f"aggregation={str(self.aggregation)}")
        return f"[{', '.join(metadata_list)}]"
