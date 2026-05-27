from typing import Any

from neuralogic.core.constructs import relation
from neuralogic.core.constructs.predicate import Predicate
from neuralogic.core.constructs.term import Constant, Variable


class SpecialPredicateFactory:
    """
    Factory for creating special predicates, such as 'alldiff', 'neq', 'eq', etc.
    Special predicates are handled differently by the backend engine.
    """

    def __init__(self, hidden: bool = False):
        """
        Parameters
        ----------
        hidden : bool
            Whether the created predicates should be hidden. Default: False.
        """
        self.is_hidden = hidden

    @property
    def hidden(self) -> "SpecialPredicateFactory":
        return SpecialPredicateFactory(True)

    def alldiff(self, *args: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("alldiff", len(args), self.is_hidden, True), args)

    def neq(self, a: Any, b: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("neq", 2, self.is_hidden, True), [a, b])

    def eq(self, a: Any, b: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("eq", 2, self.is_hidden, True), [a, b])

    def leq(self, a: Any, b: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("leq", 2, self.is_hidden, True), [a, b])

    def lt(self, a: Any, b: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("lt", 2, self.is_hidden, True), [a, b])

    def geq(self, a: Any, b: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("geq", 2, self.is_hidden, True), [a, b])

    def gt(self, a: Any, b: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("gt", 2, self.is_hidden, True), [a, b])

    def next(self, a: Any, b: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("next", 2, self.is_hidden, True), [a, b])

    def maxcard(self, *args: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("maxcard", len(args), self.is_hidden, True), args)

    def _in(self, *args: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("in", len(args), self.is_hidden, True), args)

    def anypred(self) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("anypred", 0, self.is_hidden, True), None)

    def truepred(self) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("truepred", 0, self.is_hidden, True), None)

    def add(self, a: Any, b: Any, c: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("add", 3, self.is_hidden, True), [a, b, c])

    def sub(self, a: Any, b: Any, c: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("sub", 3, self.is_hidden, True), [a, b, c])

    def mod(self, a: Any, b: Any, c: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("mod", 3, self.is_hidden, True), [a, b, c])

    def add_eval(self, a: Any, b: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("add_eval", 2, self.is_hidden, True), [a, b])

    def sub_eval(self, a: Any, b: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("sub_eval", 2, self.is_hidden, True), [a, b])

    def mod_eval(self, a: Any, b: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("mod_eval", 2, self.is_hidden, True), [a, b])

    def mul_eval(self, a: Any, b: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("mul_eval", 2, self.is_hidden, True), [a, b])

    def div_eval(self, a: Any, b: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("div_eval", 2, self.is_hidden, True), [a, b])

    def max_eval(self, a: Any, b: Any) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate("max_eval", 2, self.is_hidden, True), [a, b])

    def min_eval(self, a: Any, b: Any) -> relation.BaseRelation:
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
        """
        Creates a special relation with the given name and 0 arity.

        Parameters
        ----------
        name : str
            The name of the special predicate.

        Returns
        -------
        relation.BaseRelation
            The created special relation.
        """
        return relation.BaseRelation(Predicate(name, 0, self.is_hidden, True))


class HiddenPredicateFactory:
    """
    Factory for creating hidden predicates.
    Hidden predicates are not part of the output unless explicitly requested.
    """

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
    """
    Factory for creating atoms (relations) in the logic program.
    It supports dot notation for predicate names and provides access to special and hidden factories.
    """

    def __init__(self):
        """Initializes the AtomFactory."""
        self.instances: dict[str, dict[int, relation.BaseRelation]] = {}

        self.special = SpecialPredicateFactory()
        self.hidden = HiddenPredicateFactory()

    def get(self, name: str) -> relation.BaseRelation:
        """
        Creates a relation with the given name and 0 arity.

        Parameters
        ----------
        name : str
            The name of the predicate.

        Returns
        -------
        relation.BaseRelation
            The created relation.
        """
        return relation.BaseRelation(Predicate(name, 0, False, False))

    def __getattr__(self, item) -> relation.BaseRelation:
        return relation.BaseRelation(Predicate(item, 0, False, False))

    @staticmethod
    def get_predicate(name: str, arity: int, hidden: bool, special: bool) -> Predicate:
        return Predicate(name, arity, hidden, special)


class VariableFactory:
    """
    Factory for creating variables.
    Variables are automatically capitalized unless specified otherwise.
    """

    def __getattr__(self, item: str) -> Variable:
        return self.get(item)

    def get(self, item: str, var_type: str | None = None) -> Variable:
        """
        Creates a variable with the given name and optional type.

        Parameters
        ----------
        item : str
            The name of the variable.
        var_type : str, optional
            The type of the variable. Default: None.

        Returns
        -------
        Variable
            The created variable.
        """
        return Variable(item.capitalize(), var_type)


class ConstantFactory:
    """
    Factory for creating constants.
    Constants are automatically converted to lowercase unless specified otherwise.
    """

    def __getattr__(self, item: str) -> Constant:
        return self.get(item)

    def get(self, item: str, const_type: str | None = None) -> Constant:
        """
        Creates a constant with the given name and optional type.

        Parameters
        ----------
        item : str
            The name of the constant.
        const_type : str, optional
            The type of the constant. Default: None.

        Returns
        -------
        Constant
            The created constant.
        """
        return Constant(item.lower(), const_type)


Var = VariableFactory()
Relation = AtomFactory()
Const = ConstantFactory()

V = Var
C = Const
R = Relation
