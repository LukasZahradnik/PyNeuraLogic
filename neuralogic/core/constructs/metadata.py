from typing import Union, Iterable, Callable
from neuralogic.core.constructs.function.tree import FunctionalTree

from neuralogic.core.constructs.function import Transformation, Combination, Aggregation, Function


class Metadata:
    __slots__ = "learnable", "transformation", "aggregation", "duplicit_grounding", "combination"

    def __init__(
        self,
        learnable: bool = None,
        transformation: Union[str, Transformation, Combination] = None,
        combination: Union[str, Combination] = None,
        aggregation: Union[str, Aggregation] = None,
        duplicit_grounding: bool = False,
    ):
        self.learnable = learnable
        self.combination = combination
        self.transformation = transformation
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
            elif isinstance(entry, Transformation):
                # identity is transformation
                metadata.transformation = entry
            elif isinstance(entry, Combination):
                # sum is combination
                metadata.combination = entry
            else:
                raise ValueError(f"Invalid entry for metadata: {entry}")
        return metadata
    

    def __str__(self):
        metadata_list = []
        if self.learnable is not None:
            metadata_list.append(f"learnable={str(self.learnable).lower()}")
        if self.transformation is not None:
            metadata_list.append(f"transformation={str(self.transformation)}")
        if self.combination is not None:
            metadata_list.append(f"combination={str(self.combination)}")
        if self.aggregation is not None:
            metadata_list.append(f"aggregation={str(self.aggregation)}")
        return f"[{', '.join(metadata_list)}]"

    def copy(self) -> "Metadata":
        return Metadata(
            self.learnable, self.transformation, self.combination, self.aggregation, self.duplicit_grounding
        )
