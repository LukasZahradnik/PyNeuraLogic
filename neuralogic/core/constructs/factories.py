from typing import Dict, Optional
from neuralogic.core.constructs.predicate import Predicate
from neuralogic.core.constructs import relation
from neuralogic.core.constructs.term import Constant, Variable


class SpecialPredicateFactory:
    def __init__(self, hidden: bool = False):
        self.is_hidden = hidden

    @property
    def hidden(self):
        return SpecialPredicateFactory(True)

    def alldiff(self, *args) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("alldiff", len(args), self.is_hidden, True), args)

    def neq(self, a, b) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("neq", 2, self.is_hidden, True), [a, b])

    def eq(self, a, b) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("eq", 2, self.is_hidden, True), [a, b])

    def leq(self, a, b) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("leq", 2, self.is_hidden, True), [a, b])

    def lt(self, a, b) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("lt", 2, self.is_hidden, True), [a, b])

    def geq(self, a, b) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("geq", 2, self.is_hidden, True), [a, b])

    def gt(self, a, b) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("gt", 2, self.is_hidden, True), [a, b])

    def next(self, a, b) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("next", 2, self.is_hidden, True), [a, b])

    def maxcard(self, *args) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("maxcard", len(args), self.is_hidden, True), args)

    def _in(self, *args) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("in", len(args), self.is_hidden, True), args)

    def anypred(self) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("anypred", 0, self.is_hidden, True), None)

    def truepred(self) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("truepred", 0, self.is_hidden, True), None)

    def add(self, a, b, c) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("add", 3, self.is_hidden, True), [a, b, c])

    def sub(self, a, b, c) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("sub", 3, self.is_hidden, True), [a, b, c])

    def mod(self, a, b, c) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("mod", 3, self.is_hidden, True), [a, b, c])

    def add_eval(self, a, b) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("add_eval", 2, self.is_hidden, True), [a, b])

    def sub_eval(self, a, b) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("sub_eval", 2, self.is_hidden, True), [a, b])

    def mod_eval(self, a, b) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("mod_eval", 2, self.is_hidden, True), [a, b])

    def mul_eval(self, a, b) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("mul_eval", 2, self.is_hidden, True), [a, b])

    def div_eval(self, a, b) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("div_eval", 2, self.is_hidden, True), [a, b])

    def max_eval(self, a, b) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("max_eval", 2, self.is_hidden, True), [a, b])

    def min_eval(self, a, b) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("min_eval", 2, self.is_hidden, True), [a, b])

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

    def __getattr__(self, item):
        return relation.BaseRelation(Predicate(item, 0, self.is_hidden, True))

    def get(self, name: str) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate(name, 0, self.is_hidden, True))


class HiddenPredicateFactory:
    @property
    def special(self) -> SpecialPredicateFactory:
        return SpecialPredicateFactory(hidden=True)

    def __getattr__(self, item):
        return relation.BaseRelation(Predicate(item, 0, True, False))

    def get(self, name: str) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate(name, 0, True, False))

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


class AtomFactory:
    def __init__(self):
        self.instances: Dict[str, Dict[int, relation.BaseRelation]] = {}

        self.special = SpecialPredicateFactory()
        self.hidden = HiddenPredicateFactory()

    def get(self, name: str) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate(name, 0, False, False))

    def __getattr__(self, item) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate(item, 0, False, False))

    @staticmethod
    def get_predicate(name, arity, hidden, special) -> Predicate:
        return Predicate(name, arity, hidden, special)


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
