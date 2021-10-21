from typing import Iterable, Union

from neuralogic.core.constructs.predicate import Predicate
from neuralogic.core.constructs import rule, factories


AtomType = Union["BaseAtom", "WeightedAtom"]
BodyAtomType = Union["BaseAtom", "WeightedAtom"]

Head = AtomType
Body = Union[Iterable[BodyAtomType], BodyAtomType]


class BaseAtom:
    def __init__(self, predicate: Predicate, terms=None, negated=False):
        self.predicate = predicate
        self.negated = negated
        self.terms = terms

        if self.terms is None:
            self.terms = []
        elif not isinstance(self.terms, Iterable):
            self.terms = [self.terms]

    def __neg__(self) -> "BaseAtom":
        return self.__invert__()

    def __invert__(self) -> "BaseAtom":
        return BaseAtom(self.predicate, self.terms, not self.negated)

    def __truediv__(self, other):
        if not isinstance(other, int) or self.predicate.arity != 0 or other < 0:
            raise NotImplementedError

        name, private, special = self.predicate.name, self.predicate.private, self.predicate.special
        return factories.AtomFactory.Predicate.get_predicate(name, other, private, special)

    def __call__(self, *args) -> "BaseAtom":
        if self.terms:
            raise Exception

        terms = list(args)
        arity = len(terms)

        name, private, special = self.predicate.name, self.predicate.private, self.predicate.special
        predicate = factories.AtomFactory.Predicate.get_predicate(name, arity, private, special)

        return BaseAtom(predicate, terms, self.negated)

    def __getitem__(self, item) -> "WeightedAtom":
        # if self.java_object is None:
        #     raise NotImplementedError
        return WeightedAtom(self, item)

    def __le__(self, other: Body) -> rule.Rule:
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
        atom = BaseAtom.__new__(BaseAtom)
        atom.negated = self.negated
        atom.terms = self.terms
        atom.predicate = self.predicate
        # atom.java_object = self.java_object


class WeightedAtom:  # todo gusta: mozna dedeni namisto kompozice?
    def __init__(self, atom: BaseAtom, weight, fixed=False):
        self.atom = atom

        self.weight = weight
        self.weight_name = None
        self.is_fixed = fixed

        if isinstance(weight, slice):
            self.weight_name = str(weight.start)
            self.weight = weight.stop
        elif isinstance(weight, tuple) and isinstance(weight[0], slice):
            self.weight_name = str(weight[0].start)
            self.weight = (weight[0].stop, *weight[1:])

        if isinstance(weight, Iterable) and not isinstance(weight, tuple):
            self.weight = list(weight)

    def fixed(self) -> "WeightedAtom":
        if self.is_fixed:
            raise Exception

        # set_field(get_field(self.java_object, "weight"), "isFixed", True)
        return WeightedAtom(self.atom, self.weight, True)

    @property
    def negated(self):
        return self.atom.negated

    @property
    def predicate(self):
        return self.atom.predicate

    @property
    def terms(self):  # todo gusta: ...tim bys usetril toto volani atp.
        return self.atom.terms

    def __invert__(self) -> "WeightedAtom":
        return WeightedAtom(~self.atom, self.weight, self.is_fixed)

    def __neg__(self) -> "WeightedAtom":
        return self.__invert__()

    def __le__(self, other: Body) -> rule.Rule:
        return rule.Rule(self, other)

    def to_str(self, end=False):
        if isinstance(self.weight, tuple):
            weight = f"{{{', '.join(str(w) for w in self.weight)}}}"
        else:
            weight = str(self.weight)

        if self.is_fixed:
            return f"<{weight}> {self.atom.to_str(end)}"
        return f"{weight} {self.atom.to_str(end)}"

    def __str__(self):
        return self.to_str(True)

    def __copy__(self):
        atom = WeightedAtom.__new__(WeightedAtom)
        atom.atom = self.atom
        atom.weight = self.weight
        atom.is_fixed = self.is_fixed
        atom.java_object = self.java_object
