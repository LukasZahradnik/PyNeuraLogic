from neuralogic.core.constructs.metadata import Metadata
from typing import Iterable, Optional


class Rule:
    def __init__(self, head, body):
        from neuralogic.core import Relation

        self.head = head

        if not isinstance(body, Iterable):
            body = [body]

        self.body = list(body)
        self.metadata: Optional[Metadata] = None

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

    def __str__(self):
        metadata = "" if self.metadata is None is None else f" {self.metadata}"
        return f"{self.head.to_str()} :- {', '.join(atom.to_str() for atom in self.body)}.{metadata}"

    def __and__(self, other):
        if isinstance(other, Iterable):
            self.body.extend(list(other))
        else:
            self.body.append(other)
        return self

    def __or__(self, other):
        if not isinstance(other, Metadata):
            raise NotImplementedError
        self.metadata = other

        return self
