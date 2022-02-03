from typing import Sequence

from neuralogic.core.enums import Activation, Aggregation, ActivationAggregation, ActivationAgg
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
        if isinstance(other, Sequence):
            metadata = Metadata()

            for entry in other:
                if isinstance(entry, (Activation, ActivationAgg, ActivationAggregation)):
                    metadata.activation = entry
                elif isinstance(entry, Aggregation):
                    metadata.aggregation = entry
                else:
                    raise NotImplementedError
            other = metadata
        elif not isinstance(other, Metadata):
            raise NotImplementedError
        return PredicateMetadata(self, other)


class PredicateMetadata:
    def __init__(self, predicate: Predicate, metadata: Metadata):
        if metadata.aggregation is not None or metadata.learnable is not None:
            raise NotImplementedError

        self.predicate = predicate
        self.metadata = metadata

    def __str__(self):
        return f"{self.predicate} {self.metadata}"
