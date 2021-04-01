from neuralogic.model.java_objects import get_java_factory
from typing import Iterable


class Rule:
    def __init__(self, head, body):
        self.head = head

        if not isinstance(body, Iterable):
            body = [body]
        self.body = list(body)

        self.java_object = get_java_factory().get_rule(self)

    def __str__(self):
        return f"{self.head.to_str()} :- {', '.join(atom.to_str() for atom in self.body)}."

    def __and__(self, other):
        if isinstance(other, Iterable):
            self.body.extend(list(other))
        else:
            self.body.append(other)
        return self
