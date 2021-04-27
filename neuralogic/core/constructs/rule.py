from neuralogic.core.constructs.java_objects import get_java_factory
from neuralogic.core.constructs.metadata import Metadata
from typing import Iterable, Optional


class Rule:
    def __init__(self, head, body):
        self.head = head

        if not isinstance(body, Iterable):
            body = [body]

        self.body = list(body)
        self.metadata: Optional[Metadata] = None

        self.java_object = get_java_factory().get_rule(self)

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
        self.java_object.setMetadata(get_java_factory().get_metadata(self.metadata))

        return self
