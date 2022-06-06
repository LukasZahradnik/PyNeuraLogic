from typing import Iterable, Union

import numpy as np

from neuralogic.core.constructs.predicate import Predicate
from neuralogic.core.constructs import rule, factories


class BaseRelation:
    __slots__ = "predicate", "negated", "terms"

    def __init__(self, predicate: Predicate, terms=None, negated=False):
        self.predicate = predicate
        self.negated = negated
        self.terms = terms

        if self.terms is None:
            self.terms = []
        elif not isinstance(self.terms, Iterable):
            self.terms = [self.terms]

    def __neg__(self) -> "BaseRelation":
        return self.__invert__()

    def __invert__(self) -> "BaseRelation":
        return BaseRelation(self.predicate, self.terms, not self.negated)

    def __truediv__(self, other):
        if not isinstance(other, int) or self.predicate.arity != 0 or other < 0:
            raise NotImplementedError

        name, hidden, special = self.predicate.name, self.predicate.hidden, self.predicate.special
        return factories.AtomFactory.Predicate.get_predicate(name, other, hidden, special)

    def __call__(self, *args) -> "BaseRelation":
        if self.terms:
            raise Exception

        if len(args) == 1 and isinstance(args[0], Iterable) and not isinstance(args[0], str):
            terms = list(args[0])
        else:
            terms = list(args)
        arity = len(terms)

        name, hidden, special = self.predicate.name, self.predicate.hidden, self.predicate.special
        predicate = factories.AtomFactory.Predicate.get_predicate(name, arity, hidden, special)

        return BaseRelation(predicate, terms, self.negated)

    def __getitem__(self, item) -> "WeightedRelation":
        return WeightedRelation(item, self.predicate, False, self.terms, self.negated)

    def __le__(self, other: Union[Iterable["BaseRelation"], "BaseRelation"]) -> rule.Rule:
        return rule.Rule(self, other)

    def to_str(self, end=False) -> str:
        negation = "~" if self.negated else ""
        end = "." if end else ""

        if self.terms:
            terms = ", ".join(str(term) for term in self.terms)
            return f"{negation}{self.predicate.to_str()}({terms}){end}"
        return f"{negation}{self.predicate.to_str()}{end}"

    def __str__(self) -> str:
        return self.to_str(True)

    def __copy__(self):
        atom = BaseRelation.__new__(BaseRelation)
        atom.negated = self.negated
        atom.terms = self.terms
        atom.predicate = self.predicate

        return atom


class WeightedRelation(BaseRelation):
    __slots__ = "weight", "weight_name", "is_fixed"

    def __init__(self, weight, predicate: Predicate, fixed=False, terms=None, negated=False):
        super().__init__(predicate, terms, negated)

        self.weight = weight
        self.weight_name = None
        self.is_fixed = fixed

        if isinstance(weight, slice):
            self.weight_name = str(weight.start)
            self.weight = weight.stop
        elif isinstance(weight, tuple) and isinstance(weight[0], slice):
            self.weight_name = str(weight[0].start)
            self.weight = (weight[0].stop, *weight[1:])

        if isinstance(weight, np.ndarray):
            self.weight = weight.tolist()
        elif isinstance(weight, Iterable) and not isinstance(weight, tuple):
            self.weight = list(weight)

    def fixed(self) -> "WeightedRelation":
        if self.is_fixed:
            raise Exception(f"Weighted relation is already fixed")
        return WeightedRelation(self.weight, self.predicate, True, self.terms, self.negated)

    def __invert__(self) -> "WeightedRelation":
        return WeightedRelation(self.weight, self.predicate, self.is_fixed, self.terms, not self.negated)

    def __neg__(self) -> "WeightedRelation":
        return self.__invert__()

    def to_str(self, end=False):
        if isinstance(self.weight, tuple):
            weight = f"{{{', '.join(str(w) for w in self.weight)}}}"
        else:
            weight = str(self.weight)
        if self.weight_name:
            weight = f"${self.weight_name}={weight}"

        if self.is_fixed:
            return f"<{weight}> {super().to_str(end)}"
        return f"{weight} {super().to_str(end)}"

    def __str__(self):
        return self.to_str(True)

    def __call__(self, *args) -> None:
        raise NotImplementedError(f"Cannot assign terms to weighted relation {self.predicate}")

    def __getitem__(self, item) -> None:
        raise NotImplementedError(f"Cannot assign weight to weighted relation {self.predicate}")

    def __copy__(self):
        relation = WeightedRelation.__new__(WeightedRelation)

        relation.predicate = self.predicate
        relation.negated = self.negated
        relation.terms = self.terms
        relation.weight = self.weight
        relation.is_fixed = self.is_fixed

        return relation
