from typing import Iterable, Union

import numpy as np

from neuralogic.core.constructs.predicate import Predicate
from neuralogic.core.constructs import rule, factories
from neuralogic.core.constructs.function import Transformation, Combination


class BaseRelation:
    __slots__ = "predicate", "function", "terms", "negated"

    def __init__(
        self,
        predicate: Predicate,
        terms=None,
        function: Union[Transformation, Combination] = None,
        negated: bool = False,
    ):
        self.predicate = predicate
        self.function = function
        self.negated = negated
        self.terms = []

        if not isinstance(terms, Iterable) or isinstance(terms, str):
            terms = [terms]

        for term in terms:
            if term is None:
                continue

            if isinstance(term, list):
                self.terms.extend(term)
            else:
                self.terms.append(term)

    def __neg__(self) -> "BaseRelation":
        return self.attach_activation_function(Transformation.REVERSE)

    def __invert__(self) -> "BaseRelation":
        if self.function is not None:
            raise ValueError(f"Cannot negate relation {self} with attached function.")

        predicate = Predicate(self.predicate.name, self.predicate.arity, True, self.predicate.special)
        relation = BaseRelation(predicate, self.terms, self.function, not self.negated)

        return relation

    @property
    def T(self) -> "BaseRelation":
        return self.attach_activation_function(Transformation.TRANSP)

    def attach_activation_function(self, function: Union[Transformation, Combination]):
        if self.negated:
            raise ValueError(f"Cannot attach function to negated relation {self}")
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

        return BaseRelation(predicate, terms, self.function, self.negated)

    def __getitem__(self, item) -> "WeightedRelation":
        if self.predicate.hidden or self.predicate.special:
            raise ValueError(f"Special/Hidden relation {self} cannot have learnable parameters.")
        return WeightedRelation(item, self.predicate, False, self.terms, self.function)

    def __le__(self, other: Union[Iterable["BaseRelation"], "BaseRelation"]) -> rule.Rule:
        return rule.Rule(self, other)

    def to_str(self, end=False) -> str:
        end = "." if end else ""

        if self.terms:
            terms = ", ".join([str(term) for term in self.terms])

            if self.negated:
                return f"!{self.predicate.to_str()}({terms}){end}"
            if self.function:
                literal = f"{self.predicate.to_str()}({terms})"
                return f"{self.function.wrap(literal)}{end}"
            return f"{self.predicate.to_str()}({terms}){end}"

        if self.negated:
            return f"!{self.predicate.to_str()}{end}"
        if self.function:
            return f"{self.function.wrap(self.predicate.to_str())}{end}"
        return f"{self.predicate.to_str()}{end}"

    def __str__(self) -> str:
        return self.to_str(True)

    def __repr__(self) -> str:
        return self.__str__()

    def __copy__(self):
        relation = BaseRelation.__new__(BaseRelation)
        relation.function = self.function
        relation.terms = self.terms
        relation.predicate = self.predicate
        relation.negated = self.negated

        return relation

    def __and__(self, other) -> rule.RuleBody:
        if isinstance(other, BaseRelation):
            return rule.RuleBody(self, other)
        raise NotImplementedError


class WeightedRelation(BaseRelation):
    __slots__ = "weight", "weight_name", "is_fixed"

    def __init__(
        self, weight, predicate: Predicate, fixed=False, terms=None, function: Union[Transformation, Combination] = None
    ):
        super().__init__(predicate, terms, function, False)

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

    def __str__(self) -> str:
        return self.to_str(True)

    def __repr__(self) -> str:
        return self.__str__()

    def __call__(self, *args) -> BaseRelation:
        raise NotImplementedError(f"Cannot assign terms to weighted relation {self.predicate}")

    def __getitem__(self, item) -> "WeightedRelation":
        raise NotImplementedError(f"Cannot assign weight to weighted relation {self.predicate}")

    def attach_activation_function(self, function: Union[Transformation, Combination]):
        raise NotImplementedError(
            f"Cannot attach a function to weighted relation {self}. Attach the function before adding weights."
        )

    @property
    def T(self) -> "WeightedRelation":
        raise NotImplementedError(
            f"Cannot transpose weighted relation {self} Apply the transposition before adding weights."
        )

    def __invert__(self) -> "WeightedRelation":
        raise NotImplementedError(f"Weighted relations ({self}) cannot be negated.")

    def __neg__(self) -> "WeightedRelation":
        raise NotImplementedError(
            f"Cannot negate weighted relation {self} Apply the reverse function before adding weights."
        )

    def __copy__(self):
        relation = WeightedRelation.__new__(WeightedRelation)

        relation.predicate = self.predicate
        relation.function = self.function
        relation.terms = self.terms
        relation.weight = self.weight
        relation.is_fixed = self.is_fixed
        relation.negated = self.negated

        return relation
