from typing import Dict
from neuralogic.core.constructs.predicate import Predicate
from neuralogic.core.constructs import relation


class AtomFactory:
    class Predicate:
        def __init__(self, hidden=False, special=False):
            self.is_hidden = hidden
            self.is_special = special

        @property
        def special(self) -> "AtomFactory.Predicate":
            return AtomFactory.Predicate(self.is_hidden, True)

        @property
        def hidden(self) -> "AtomFactory.Predicate":
            return AtomFactory.Predicate(True, self.is_special)

        @staticmethod
        def get_predicate(name, arity, hidden, special) -> Predicate:
            return Predicate(name, arity, hidden, special)

        def __getattr__(self, item):
            return relation.BaseRelation(Predicate(item, 0, self.is_hidden, self.is_special))

        def get(self, name: str) -> relation.BaseRelation:
            return relation.BaseRelation(Predicate(name, 0, self.is_hidden, self.is_special))

        def __call__(self, *args, **kwargs):
            raise Exception(
                "Cannot add terms to not fully initialized relation - 'special' and 'hidden' are keywords, "
                "that cannot be used as a predicate name with dot notation (use `get` method instead)"
            )

        def __getitem__(self, item):
            raise Exception(
                "Cannot add terms to not fully initialized relation - 'special' and 'hidden' are keywords, "
                "that cannot be used as a predicate name with dot notation (use `get` method instead)"
            )

    def __init__(self):
        self.instances: Dict[str, Dict[int, relation.BaseRelation]] = {}

        self.special = AtomFactory.Predicate(special=True)
        self.hidden = AtomFactory.Predicate(hidden=True)

    def get(self, name: str) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate(name, 0, False, False))

    def __getattr__(self, item) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate(item, 0, False, False))


class VariableFactory:
    def __getattr__(self, item: str) -> str:
        return item.capitalize()


class ConstantFactory:
    def __getattr__(self, item: str) -> str:
        return item.lower()


Var = VariableFactory()
Relation = AtomFactory()
Constant = ConstantFactory()

V = Var
C = Constant
R = Relation
