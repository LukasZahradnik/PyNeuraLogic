from typing import Dict, Optional
from neuralogic.core.constructs.predicate import Predicate
from neuralogic.core.constructs import relation
from neuralogic.core.constructs.term import Constant, Variable


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
    def __getattr__(self, item: str) -> Variable:
        return self.get(item)

    def get(self, item: str, var_type: Optional[str] = None) -> Variable:
        return Variable(item.capitalize(), var_type)


class ConstantFactory:
    def __getattr__(self, item: str) -> Constant:
        return self.get(item)

    def get(self, item: str, const_type: Optional[str] = None) -> Constant:
        return Constant(item.lower(), const_type)


Var = VariableFactory()
Relation = AtomFactory()
Const = ConstantFactory()

V = Var
C = Const
R = Relation
