from typing import Union, Iterable, Callable, Optional

from neuralogic.core.constructs.function import Transformation, Combination, Aggregation, Function


class Metadata:
    __slots__ = "learnable", "transformation", "aggregation", "duplicate_grounding", "combination"

    def __init__(
        self,
        learnable: bool = None,
        transformation: Union[str, Transformation, Combination] = None,
        combination: Union[str, Combination] = None,
        aggregation: Union[str, Aggregation] = None,
        duplicate_grounding: Optional[bool] = None,
    ):
        self.learnable = learnable
        self.combination = combination
        self.transformation = transformation
        self.aggregation = aggregation
        self.duplicate_grounding = duplicate_grounding

    @staticmethod
    def from_iterable(iterable: Iterable) -> "Metadata":
        metadata = Metadata()

        for entry in iterable:
            if isinstance(entry, Callable) and not isinstance(entry, Function):
                entry = entry()
            if isinstance(entry, Aggregation):
                metadata.aggregation = entry
            elif isinstance(entry, Transformation):
                metadata.transformation = entry
            elif isinstance(entry, Combination):
                metadata.combination = entry
            else:
                raise ValueError(f"Invalid entry for metadata: {entry}")
        return metadata

    def __str__(self) -> str:
        metadata_list = []
        if self.learnable is not None:
            metadata_list.append(f"learnable={str(self.learnable).lower()}")
        if self.transformation is not None:
            metadata_list.append(f"transformation={str(self.transformation)}")
        if self.combination is not None:
            metadata_list.append(f"combination={str(self.combination)}")
        if self.aggregation is not None:
            metadata_list.append(f"aggregation={str(self.aggregation)}")
        if self.duplicate_grounding is not None:
            metadata_list.append(f"duplicate_grounding={str(self.duplicate_grounding)}")
        return f"[{', '.join(metadata_list)}]"

    def __repr__(self) -> str:
        return self.__str__()

    def combine(self, other: "Metadata") -> "Metadata":
        return Metadata(
            learnable=other.learnable if other.learnable is not None else self.learnable,
            transformation=other.transformation if other.transformation is not None else self.transformation,
            combination=other.combination if other.combination is not None else self.combination,
            aggregation=other.aggregation if other.aggregation is not None else self.aggregation,
            duplicit_grounding=other.duplicit_grounding
            if other.duplicit_grounding is not None
            else self.duplicit_grounding,
        )

    def __add__(self, other: "Metadata") -> "Metadata":
        return self.combine(other)

    def copy(self) -> "Metadata":
        return Metadata(
            learnable=self.learnable,
            transformation=self.transformation,
            combination=self.combination,
            aggregation=self.aggregation,
            duplicate_grounding=self.duplicate_grounding,
        )
