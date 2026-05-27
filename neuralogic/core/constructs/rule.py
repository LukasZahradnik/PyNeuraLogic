from typing import Iterable, Any, Union, TypeAlias

from neuralogic.core.constructs.function import FContainer
from neuralogic.core.constructs.metadata import Metadata


BodyItem: TypeAlias = Union["BaseRelation", FContainer]


class RuleBody:
    """
    Represents the body of a rule, which is a collection of literals.
    """

    __slots__ = "literals", "metadata"

    def __init__(self, lit1: "BaseRelation", lit2: "BaseRelation"):
        """
        Parameters
        ----------
        lit1 : BaseRelation
            The first literal in the body.
        lit2 : BaseRelation
            The second literal in the body.
        """
        self.literals: list[Union["BaseRelation", FContainer]] = [lit1, lit2]
        self.metadata: Metadata | None = None

    def __and__(self, other: Union["BaseRelation", FContainer]) -> "RuleBody":
        from neuralogic.core.constructs.relation import BaseRelation

        if isinstance(other, (BaseRelation, FContainer)):
            self.literals.append(other)
            return self
        raise NotImplementedError

    def __str__(self) -> str:
        return ", ".join(atom.to_str() for atom in self.literals)

    def __repr__(self) -> str:
        return self.__str__()

    def __or__(self, other: Iterable | Metadata) -> "RuleBody":
        if isinstance(other, Iterable):
            other = Metadata.from_iterable(other)
        elif not isinstance(other, Metadata):
            raise NotImplementedError

        self.metadata = other
        return self


class Rule:
    """
    Represents a rule in the model, consisting of a head and a body.
    """

    __slots__ = "head", "body", "metadata"

    def __init__(self, head: "BaseRelation", body: RuleBody | Iterable[BodyItem] | BodyItem):
        """
        Parameters
        ----------
        head : BaseRelation
            The head of the rule.
        body : RuleBody | Iterable[BaseRelation] | BaseRelation
            The body of the rule.
        """
        self.head = head
        self.metadata: Metadata | None = None

        if head.function is not None:
            raise NotImplementedError(f"Rule head {head} cannot have a function attached")

        if isinstance(body, RuleBody):
            if body.metadata is not None:
                self._set_metadata(body.metadata)
            body = body.literals

        if not isinstance(body, Iterable):
            body = [body]

        self.body: list[BodyItem] | FContainer = body

        if not isinstance(self.body, FContainer):
            self.body = list(self.body)

    def _contains_function_container(self):
        for lit in self.body:
            if isinstance(lit, FContainer):
                return True
        return False

    def to_str(self, _: bool = False) -> str:
        return str(self)

    def __str__(self) -> str:
        metadata = "" if self.metadata is None is None else f" {self.metadata}"
        if isinstance(self.body, FContainer):
            return f"{self.head.to_str()} :- {self.body.to_str()}.{metadata}"
        return f"{self.head.to_str()} :- {', '.join(atom.to_str() for atom in self.body)}.{metadata}"

    def __repr__(self) -> str:
        return self.to_str()

    def __and__(self, other: Iterable[BodyItem] | BodyItem) -> "Rule":
        if isinstance(other, Iterable):
            self.body.extend(list(other))
        else:
            self.body.append(other)
        return self

    def __or__(self, other: Iterable | Metadata) -> "Rule":
        if isinstance(other, Iterable):
            other = Metadata.from_iterable(other)
        elif not isinstance(other, Metadata):
            raise NotImplementedError

        self._set_metadata(other)
        return self

    def _set_metadata(self, metadata: Metadata):
        if metadata.aggregation is not None and metadata.aggregation.rule_head_dependant():
            metadata = metadata.copy()
            metadata.aggregation = metadata.aggregation.process_head(self.head)

        self.metadata = metadata
