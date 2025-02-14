from typing import Optional

import jpype


class Function:
    __slots__ = "name", "operator", "can_flatten", "namespace"

    def __init__(self, name: str, *, namespace: str = "", operator: Optional[str] = None, can_flatten: bool = False):
        self.name: str = name.lower()
        self.operator: Optional[str] = operator
        self.can_flatten = can_flatten
        self.namespace = namespace

    def __str__(self):
        return self.name

    def wrap(self, content: str) -> str:
        return f"{self.name}({content})"

    def pretty_str(self) -> str:
        return str(self).capitalize()

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            return self
        raise NotImplementedError

    def is_parametrized(self) -> bool:
        return False

    def rule_head_dependant(self) -> bool:
        return False

    def process_head(self, head) -> "Function":
        raise NotImplementedError

    def get(self):
        name = "".join(s.capitalize() for s in self.name.split("_"))
        formatted_namespace = self.namespace.format(name=name)

        return jpype.JClass(f"cz.cvut.fel.ida.algebra.functions.{formatted_namespace}")()


class TransformationFunction(Function):
    def __init__(
        self,
        name: str,
        *,
        namespace: str = "transformation.elementwise.{name}",
        operator: Optional[str] = None,
        can_flatten: bool = False,
    ):
        super().__init__(name, namespace=namespace, operator=operator, can_flatten=can_flatten)

    def __call__(self, relation=None, **kwargs):
        from neuralogic.core.constructs import relation as rel
        from neuralogic.core.constructs.function.function_container import FContainer

        if relation is None:
            return self

        if isinstance(relation, rel.BaseRelation) and not isinstance(relation, rel.WeightedRelation):
            if relation.negated or relation.function is not None:
                return FContainer((relation,), self)
            return relation.attach_activation_function(self)
        return FContainer(relation, self)


class CombinationFunction(Function):
    def __init__(
        self,
        name: str,
        *,
        namespace: str = "combination.{name}",
        operator: Optional[str] = None,
        can_flatten: bool = False,
    ):
        super().__init__(name, namespace=namespace, operator=operator, can_flatten=can_flatten)

    def __call__(self, *relations):
        from neuralogic.core.constructs.function.function_container import FContainer

        if len(relations) == 0:
            return self
        return FContainer(relations, self)


class AggregationFunction(Function):
    def get(self):
        raise NotImplementedError
