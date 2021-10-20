from typing import Dict
from neuralogic.core.constructs.predicate import Predicate
from neuralogic.core.constructs import atom


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
            return atom.BaseAtom(AtomFactory.Predicate.get_predicate(item, 0, self.is_private, self.is_special))

    def __init__(self):
        self.instances: Dict[str, Dict[int, atom.BaseAtom]] = {}

        self.special = AtomFactory.Predicate(special=True)
        self.private = AtomFactory.Predicate(private=True)

    def get(self, name: str) -> atom.BaseAtom:
        return atom.BaseAtom(AtomFactory.Predicate.get_predicate(name, 0, False, False))

    def __getattr__(self, item) -> atom.BaseAtom:
        return atom.BaseAtom(AtomFactory.Predicate.get_predicate(item, 0, False, False))


class VariableFactory:
    def __getattr__(self, item) -> str:
        return item.upper()


class TermFactory:
    def __getattr__(self, item) -> str:
        return item.lower()


Var = VariableFactory()
Relation = AtomFactory()
Term = TermFactory()


V = Var
T = Term
R = Relation