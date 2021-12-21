from collections import Iterable

from neuralogic.core.enums import Activation, Aggregation
from neuralogic.core.constructs.metadata import Metadata


class Predicate:
    """WeightedPredicate"""

    def __init__(self, name, arity, hidden=False, special=False):
        if name.startswith("_"):
            name = name[1:]
            hidden = True

        self.name = name
        self.arity = arity
        self.hidden = hidden
        self.special = special

    def set_arity(self, arity):
        if self.arity == arity:
            return self

    def to_str(self):
        special = "@" if self.special else ""
        hidden = "*" if self.hidden else ""
        return f"{hidden}{special}{self.name}"

    def __str__(self):
        special = "@" if self.special else ""
        hidden = "*" if self.hidden else ""
        return f"{hidden}{special}{self.name}/{self.arity}"

    def __or__(self, other) -> "PredicateMetadata":
        if isinstance(other, Iterable):
            metadata = Metadata()

            for entry in other:
                if isinstance(entry, Activation):
                    metadata.activation = entry
                elif isinstance(entry, Aggregation):
                    metadata.aggregation = entry
                else:
                    raise NotImplementedError
            other = metadata
        elif not isinstance(other, Metadata):
            raise NotImplementedError

        if other.aggregation is not None:
            if other.aggregation not in (Aggregation.MAX, Aggregation.MIN):
                raise NotImplementedError
            activation = Activation.IDENTITY.value if other.activation is None else other.activation.value
            other = Metadata(activation=f"{other.aggregation.value}-{activation.lower()}")

        return PredicateMetadata(self, other)


class PredicateMetadata:
    def __init__(self, predicate: Predicate, metadata: Metadata):
        if metadata.aggregation is not None or metadata.learnable is not None:
            raise NotImplementedError

        self.predicate = predicate
        self.metadata = metadata

    def __str__(self):
        return f"{self.predicate} {self.metadata}"
