from collections.abc import Sequence
from typing import Optional

from neuralogic.core.constructs.metadata import Metadata


class Predicate:
    """
    Represents a predicate in the logic program, defined by its name and arity.
    """
    __slots__ = "name", "arity", "hidden", "special"

    def __init__(self, name: str, arity: int, hidden: bool = False, special: bool = False):
        """
        Parameters
        ----------
        name : str
            The name of the predicate.
        arity : int
            The number of arguments the predicate takes.
        hidden : bool
            Whether the predicate is hidden. Default: False.
        special : bool
            Whether the predicate is a special backend predicate. Default: False.
        """
        if name.startswith("_"):
            name = name[1:]
            hidden = True

        self.name = name
        self.arity = arity
        self.hidden = hidden
        self.special = special

    def set_arity(self, arity: int) -> Optional["Predicate"]:
        if self.arity == arity:
            return self

    def to_str(self) -> str:
        """
        Returns a string representation of the predicate (without arity).

        Returns
        -------
        str
            The string representation.
        """
        if not self.special and not self.hidden:
            return self.name

        special = "@" if self.special else ""
        hidden = "*" if self.hidden else ""
        return f"{hidden}{special}{self.name}"

    def __str__(self) -> str:
        special = "@" if self.special else ""
        hidden = "*" if self.hidden else ""
        return f"{hidden}{special}{self.name}/{self.arity}"

    def __repr__(self) -> str:
        return self.__str__()

    def __or__(self, other: Sequence | Metadata) -> "PredicateMetadata":
        if isinstance(other, Sequence):
            other = Metadata.from_iterable(other)
        elif not isinstance(other, Metadata):
            raise NotImplementedError
        return PredicateMetadata(self, other)


class PredicateMetadata:
    """
    Associates metadata with a predicate.
    """
    __slots__ = "predicate", "metadata"

    def __init__(self, predicate: Predicate, metadata: Metadata):
        """
        Parameters
        ----------
        predicate : Predicate
            The predicate to associate metadata with.
        metadata : Metadata
            The metadata to associate.
        """
        if metadata.aggregation is not None:
            raise ValueError(f"Cannot set 'aggregation' parameter on predicate ({predicate}) metadata")

        if metadata.learnable is not None:
            raise ValueError(f"Cannot set 'learnable' parameter on predicate ({predicate}) metadata")

        self.predicate = predicate
        self.metadata = metadata

    def __str__(self) -> str:
        return f"{self.predicate} {self.metadata}"

    def __repr__(self) -> str:
        return self.__str__()
