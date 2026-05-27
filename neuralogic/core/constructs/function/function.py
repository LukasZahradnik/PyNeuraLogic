from typing import Any

import jpype


class Function:
    """
    Base class for all logic functions (transformation, combination, aggregation).
    Functions are used to transform, combine, or aggregate values in the logic program.
    """

    __slots__ = "name", "operator", "can_flatten", "namespace"

    def __init__(self, name: str, *, namespace: str = "", operator: str | None = None, can_flatten: bool = False):
        """
        Parameters
        ----------
        name : str
            The name of the function.
        namespace : str
            The Java namespace of the function. Default: "".
        operator : str, optional
            The operator associated with the function (e.g., '+', '@'). Default: None.
        can_flatten : bool
            Whether the function can be flattened in the logic program. Default: False.
        """
        self.name: str = name.lower()
        self.operator: str | None = operator
        self.can_flatten = can_flatten
        self.namespace = namespace

    def __str__(self) -> str:
        return self.name

    def wrap(self, content: str) -> str:
        return f"{self.name}({content})"

    def pretty_str(self) -> str:
        return str(self).capitalize()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 0:
            return self
        raise NotImplementedError

    def is_parametrized(self) -> bool:
        return False

    def rule_head_dependant(self) -> bool:
        return False

    def process_head(self, head: Any) -> "Function":
        raise NotImplementedError

    def get(self) -> Any:
        """
        Returns the Java representation of the function.

        Returns
        -------
        Any
            The Java function object.
        """
        name = "".join(s.capitalize() for s in self.name.split("_"))
        formatted_namespace = self.namespace.format(name=name)

        return jpype.JClass(f"cz.cvut.fel.ida.algebra.functions.{formatted_namespace}")()


class TransformationFunction(Function):
    """
    Represents a transformation function applied to a relation or a container of relations.
    Transformation functions can be applied element-wise or as join operations.
    """

    def __init__(
        self,
        name: str,
        *,
        namespace: str = "transformation.elementwise.{name}",
        operator: str | None = None,
        can_flatten: bool = False,
    ):
        super().__init__(name, namespace=namespace, operator=operator, can_flatten=can_flatten)

    def __call__(self, relation: Any = None, **kwargs: Any) -> Any:
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
    """
    Represents a combination function used to combine multiple relations into a single output.
    """

    def __init__(
        self,
        name: str,
        *,
        namespace: str = "combination.{name}",
        operator: str | None = None,
        can_flatten: bool = False,
    ):
        super().__init__(name, namespace=namespace, operator=operator, can_flatten=can_flatten)

    def __call__(self, *relations: Any) -> Any:
        from neuralogic.core.constructs.function.function_container import FContainer

        if len(relations) == 0:
            return self
        return FContainer(relations, self)


class AggregationFunction(Function):
    """
    Represents an aggregation function used to aggregate multiple groundings of the same rule.
    """

    def get(self) -> Any:
        raise NotImplementedError
