from typing import Iterable, Union

import numpy as np

from neuralogic.core.constructs.predicate import Predicate
from neuralogic.core.constructs import rule, factories
from neuralogic.core.constructs.function import Activation, ActivationAgg


class BaseRelation:
    __slots__ = "predicate", "function", "terms"

    def __init__(self, predicate: Predicate, terms=None, function: Union[Activation, ActivationAgg] = None):
        self.predicate = predicate
        self.function = function
        self.terms = terms

        if self.terms is None:
            self.terms = []
        elif not isinstance(self.terms, Iterable):
            self.terms = [self.terms]

    def __neg__(self) -> "BaseRelation":
        return self.__invert__()

    def __invert__(self) -> "BaseRelation":
        return self.attach_activation_function(Activation.REVERSE)

    @property
    def T(self) -> "BaseRelation":
        return self.attach_activation_function(Activation.TRANSP)

    def attach_activation_function(self, function: Union[Activation, ActivationAgg]):
        if self.function:
            function = function.nest(self.function)

        relation = self.__copy__()
        relation.function = function
        return relation

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

        return BaseRelation(predicate, terms, self.function)

    def __getitem__(self, item) -> "WeightedRelation":
        return WeightedRelation(item, self.predicate, False, self.terms, self.function)

    def __le__(self, other: Union[Iterable["BaseRelation"], "BaseRelation"]) -> rule.Rule:
        return rule.Rule(self, other)

    def to_str(self, end=False) -> str:
        end = "." if end else ""

        if self.terms:
            terms = ", ".join(str(term) for term in self.terms)
            name = f"{self.predicate.to_str()}({terms})"
        else:
            name = f"{self.predicate.to_str()}"

        if self.function:
            return f"{self.function}({name}){end}"
        return f"{name}{end}"

    def __str__(self) -> str:
        return self.to_str(True)

    def __copy__(self):
        atom = BaseRelation.__new__(BaseRelation)
        atom.function = self.function
        atom.terms = self.terms
        atom.predicate = self.predicate

        return atom


class WeightedRelation(BaseRelation):
    __slots__ = "weight", "weight_name", "is_fixed"

    def __init__(
        self, weight, predicate: Predicate, fixed=False, terms=None, function: Union[Activation, ActivationAgg] = None
    ):
        super().__init__(predicate, terms, function)

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
            raise Exception(f"Weighted relation {self} is already fixed")
        return WeightedRelation(self.weight, self.predicate, True, self.terms, self.function)

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

    def attach_activation_function(self, function: Union[Activation, ActivationAgg]):
        raise NotImplementedError(
            f"Cannot attach a function to weighted relation {self}. Attach the function before adding weights."
        )

    @property
    def T(self) -> "WeightedRelation":
        raise NotImplementedError(
            f"Cannot transpose weighted relation {self}. Apply the transposition before adding weights."
        )

    def __invert__(self) -> "WeightedRelation":
        raise NotImplementedError(f"Cannot negate weighted relation {self}. Apply the negation before adding weights.")

    def __neg__(self) -> "WeightedRelation":
        return self.__invert__()

    def __copy__(self):
        relation = WeightedRelation.__new__(WeightedRelation)

        relation.predicate = self.predicate
        relation.function = self.function
        relation.terms = self.terms
        relation.weight = self.weight
        relation.is_fixed = self.is_fixed

        return relation
