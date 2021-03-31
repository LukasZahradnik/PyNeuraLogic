from typing import Dict, Iterable, Union, Type
from neuralogic.model.predicate import Predicate


AtomType = Union["BaseAtom", "WeightedAtom"]
BodyAtomType = Union["BaseAtom", "WeightedAtom"]

Head = AtomType
Body = Union[Iterable[BodyAtomType], BodyAtomType]


class Rule:
    def __init__(self, head: Head, body: Body):
        self.head = head

        if not isinstance(body, Iterable):
            body = [body]
        self.body = list(body)

    def __str__(self):
        return f"{self.head.to_str()} :- {', '.join(atom.to_str() for atom in self.body)}."

    def __and__(self, other):
        if isinstance(other, Iterable):
            self.body.extend(list(other))
        else:
            self.body.append(other)
        return self


class BaseAtom:
    """ Atom """

    def __init__(self, predicate: "Predicate", terms=None, negated=False):
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

    def __call__(self, *args) -> "BaseAtom":
        if self.terms:
            raise Exception

        terms = list(args)
        arity = len(terms)

        name, private, special = self.predicate.name, self.predicate.private, self.predicate.special
        predicate = AtomFactory.Predicate.get_predicate(name, arity, private, special)

        return BaseAtom(predicate, terms, self.negated)

    def __getitem__(self, item) -> "WeightedAtom":
        return WeightedAtom(self, item)

    def __le__(self, other: Body) -> Rule:
        return Rule(self, other)

    def to_str(self, end=False) -> str:
        negation = "~" if self.negated else ""
        end = "." if end else ""

        if self.terms:
            terms = ", ".join(str(term) for term in self.terms)
            return f"{negation}{self.predicate.to_str()}({terms}){end}"
        return f"{negation}{self.predicate.to_str()}{end}"

    def __str__(self) -> str:
        return self.to_str(True)


class WeightedAtom:
    """ValuedFact"""

    def __init__(self, atom: BaseAtom, weight, fixed=False):
        self.atom = atom
        self.weight = weight
        self.is_fixed = fixed

        if isinstance(weight, Iterable) and not isinstance(weight, tuple):
            self.weight = list(weight)

    def fixed(self) -> "WeightedAtom":
        if self.is_fixed:
            raise Exception
        return WeightedAtom(self.atom, self.weight, True)

    @property
    def negated(self):
        return self.atom.negated

    @property
    def predicate(self):
        return self.atom.predicate

    @property
    def terms(self):
        return self.atom.terms

    def __invert__(self) -> "WeightedAtom":
        return WeightedAtom(~self.atom, self.weight, self.is_fixed)

    def __neg__(self) -> "WeightedAtom":
        return self.__invert__()

    def __le__(self, other: Body) -> Rule:
        return Rule(self, other)

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


class AtomFactory:
    class Predicate:
        predicates: Dict[str, Predicate] = {}

        def __init__(self, private=False, special=False):
            self.is_private = private
            self.is_special = special

        @property
        def special(self) -> "AtomFactory.Predicate":
            return AtomFactory.Predicate(self.is_private, True)

        @property
        def private(self) -> "AtomFactory.Predicate":
            return AtomFactory.Predicate(True, self.is_special)

        @staticmethod
        def get_predicate(name, arity, private, special) -> Predicate:
            key = f"{name}/{arity}"
            if key not in AtomFactory.Predicate.predicates:
                AtomFactory.Predicate.predicates[key] = Predicate(name, arity, private, special)
            return AtomFactory.Predicate.predicates[key]

        def __getattr__(self, item):
            return BaseAtom(AtomFactory.Predicate.get_predicate(item, 0, self.is_private, self.is_special))

    def __init__(self):
        self.instances: Dict[str, Dict[int, BaseAtom]] = {}

        self.special = AtomFactory.Predicate(special=True)
        self.private = AtomFactory.Predicate(private=True)

    def __getattr__(self, item) -> BaseAtom:
        return BaseAtom(AtomFactory.Predicate.get_predicate(item, 0, False, False))


class VariableFactory:
    def __getattr__(self, item) -> str:
        return item.upper()


class TermFactory:
    def __getattr__(self, item) -> str:
        return item.lower()


Var = VariableFactory()
Atom = AtomFactory()
Term = TermFactory()
