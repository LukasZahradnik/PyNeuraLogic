from collections.abc import Iterable
from typing import Any

from neuralogic.core.constructs.function.function import (
    Function,
    AggregationFunction,
    TransformationFunction,
    CombinationFunction,
)


class Metadata:
    """
    Represents metadata for a logic construct (e.g., rule, predicate).

    Metadata can specify properties like learnability, transformation functions,
    aggregation functions, and combination functions.
    """
    __slots__ = "learnable", "transformation", "aggregation", "duplicate_grounding", "combination"

    def __init__(
        self,
        learnable: bool | None = None,
        transformation: TransformationFunction | CombinationFunction | None = None,
        combination: CombinationFunction | None = None,
        aggregation: AggregationFunction | None = None,
        duplicate_grounding: bool | None = None,
    ):
        """
        Parameters
        ----------
        learnable : bool, optional
            Whether the construct is learnable. Default: None.
        transformation : Union[TransformationFunction, CombinationFunction], optional
            The transformation function to apply. Default: None.
        combination : CombinationFunction, optional
            The combination function to use. Default: None.
        aggregation : AggregationFunction, optional
            The aggregation function to use. Default: None.
        duplicate_grounding : bool, optional
            Whether to allow duplicate groundings. Default: None.
        """
        self.learnable = learnable
        self.combination = combination
        self.transformation = transformation
        self.aggregation = aggregation
        self.duplicate_grounding = duplicate_grounding

    @staticmethod
    def from_iterable(iterable: Iterable[Any]) -> "Metadata":
        """
        Creates a Metadata object from an iterable of functions or values.

        Parameters
        ----------
        iterable : Iterable
            The iterable containing metadata entries.

        Returns
        -------
        Metadata
            The created Metadata object.
        """
        metadata = Metadata()

        for entry in iterable:
            if callable(entry) and not isinstance(entry, Function):
                entry = entry()
            if isinstance(entry, AggregationFunction):
                metadata.aggregation = entry
            elif isinstance(entry, TransformationFunction):
                metadata.transformation = entry
            elif isinstance(entry, CombinationFunction):
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
        """
        Combines this metadata with another Metadata object.
        Values from the other object take precedence.

        Parameters
        ----------
        other : Metadata
            The other Metadata object to combine with.

        Returns
        -------
        Metadata
            A new combined Metadata object.
        """
        return Metadata(
            learnable=other.learnable if other.learnable is not None else self.learnable,
            transformation=other.transformation if other.transformation is not None else self.transformation,
            combination=other.combination if other.combination is not None else self.combination,
            aggregation=other.aggregation if other.aggregation is not None else self.aggregation,
            duplicate_grounding=other.duplicate_grounding
            if other.duplicate_grounding is not None
            else self.duplicate_grounding,
        )

    def __add__(self, other: "Metadata") -> "Metadata":
        return self.combine(other)

    def copy(self) -> "Metadata":
        """
        Returns a shallow copy of the metadata.

        Returns
        -------
        Metadata
            The copy of the metadata.
        """
        return Metadata(
            learnable=self.learnable,
            transformation=self.transformation,
            combination=self.combination,
            aggregation=self.aggregation,
            duplicate_grounding=self.duplicate_grounding,
        )
