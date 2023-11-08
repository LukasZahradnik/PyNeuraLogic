from typing import Iterable, Optional

from neuralogic.core.constructs.metadata import Metadata


class RuleBody:
    __slots__ = "literals", "metadata"

    def __init__(self, lit1, lit2):
        self.literals = [lit1, lit2]
        self.metadata = None

    def __and__(self, other):
        from neuralogic.core.constructs.relation import BaseRelation

        if isinstance(other, BaseRelation):
            self.literals.append(other)
            return self
        raise NotImplementedError

    def __str__(self) -> str:
        return ", ".join(atom.to_str() for atom in self.literals)

    def __repr__(self) -> str:
        return self.__str__()

    def __or__(self, other) -> "RuleBody":
        if isinstance(other, Iterable):
            other = Metadata.from_iterable(other)
        elif not isinstance(other, Metadata):
            raise NotImplementedError

        self.metadata = other
        return self


class Rule:
    __slots__ = "head", "body", "metadata"

    def __init__(self, head, body):
        from neuralogic.core import Relation

        self.head = head
        self.metadata: Optional[Metadata] = None

        if head.function is not None:
            raise NotImplementedError(f"Rule head {head} cannot have a function attached")

        if isinstance(body, RuleBody):
            if body.metadata is not None:
                self._set_metadata(body.metadata)
            body = body.literals

        if not isinstance(body, Iterable):
            body = [body]

        self.body = list(body)

        if self.is_ellipsis_templated():
            variable_set = {term for term in head.terms if term is not Ellipsis and str(term)[0].isupper()}

            for body_atom in self.body:
                if body_atom.predicate.special and body_atom.predicate.name == "alldiff":
                    continue

                for term in body_atom.terms:
                    if term is not Ellipsis and str(term)[0].isupper():
                        variable_set.add(term)

            for atom_index, body_atom in enumerate(self.body):
                if not body_atom.predicate.special or body_atom.predicate.name != "alldiff":
                    continue

                new_terms = []
                found_replacement = False

                for index, term in enumerate(body_atom.terms):
                    if term is Ellipsis:
                        if found_replacement:
                            raise NotImplementedError
                        found_replacement = True
                        new_terms.extend(variable_set)
                    else:
                        new_terms.append(term)
                if found_replacement:
                    self.body[atom_index] = Relation.special.alldiff(*new_terms)

    def is_ellipsis_templated(self) -> bool:
        for body_atom in self.body:
            if not body_atom.predicate.special or body_atom.predicate.name != "alldiff":
                continue
            for term in body_atom.terms:
                if term is Ellipsis:
                    return True
        return False

    def to_str(self, _: bool = False) -> str:
        return str(self)

    def __str__(self) -> str:
        metadata = "" if self.metadata is None is None else f" {self.metadata}"
        return f"{self.head.to_str()} :- {', '.join(atom.to_str() for atom in self.body)}.{metadata}"

    def __repr__(self) -> str:
        return self.to_str()

    def __and__(self, other) -> "Rule":
        if isinstance(other, Iterable):
            self.body.extend(list(other))
        else:
            self.body.append(other)
        return self

    def __or__(self, other) -> "Rule":
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
