from typing import Union, Iterable, Callable, Optional

from neuralogic.core.constructs.function import Transformation, Combination, Aggregation, Function


class Metadata:
    __slots__ = "learnable", "transformation", "aggregation", "duplicit_grounding", "combination"

    def __init__(
        self,
        learnable: bool = None,
        transformation: Union[str, Transformation, Combination] = None,
        combination: Union[str, Combination] = None,
        aggregation: Union[str, Aggregation] = None,
        duplicit_grounding: Optional[bool] = None,
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
        if self.duplicit_grounding is not None:
            metadata_list.append(f"duplicit_grounding={str(self.duplicit_grounding)}")
        return f"[{', '.join(metadata_list)}]"

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> "Metadata":
        return Metadata(
            learnable=self.learnable,
            transformation=self.transformation,
            combination=self.combination,
            aggregation=self.aggregation,
            duplicit_grounding=self.duplicit_grounding,
        )
